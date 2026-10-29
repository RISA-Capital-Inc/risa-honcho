"""Regression tests for the ``documents.content`` length cap.

``documents`` carries ``CHECK ck_documents_content_length ((length(content) <= 65535))``
while ``schemas.DocumentCreate.content`` used to accept up to 100,000
characters, so a document could pass validation at the API boundary and still
violate the table constraint.

Before the fix every document in a batch was inserted with one ``add_all`` +
``commit``; a single over-cap row raised ``IntegrityError``, the whole batch was
rolled back, and the caller turned that into one ``ValidationException`` it
swallowed. In production a deriver completion that ran to its full 16,384-token
budget produced ~65,536 characters, both observers' representations were lost,
and the queue row was marked ``processed=true`` with ``error NULL`` — a silent
loss with no counter.

The cap now lives once in ``src.constants.MAX_DOCUMENT_CONTENT_LENGTH`` and is
consumed by the CHECK constraint, the pydantic schema and the CRUD guards.
These tests pin both halves of the contract: the boundary cannot drift, and a
rejected document never takes its siblings with it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.constants import MAX_DOCUMENT_CONTENT_LENGTH

OVER_CAP = MAX_DOCUMENT_CONTENT_LENGTH + 1


async def _setup_scope(
    db_session: AsyncSession,
    test_workspace: models.Workspace,
    test_peer: models.Peer,
) -> tuple[models.Peer, models.Session]:
    """Create the observed peer, session and collection the documents need."""
    observed = models.Peer(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    db_session.add(observed)
    session = models.Session(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    db_session.add(session)
    await db_session.flush()

    db_session.add(
        models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
    )
    await db_session.flush()
    return observed, session


def _document(content: str, session: models.Session) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
        session_name=session.name,
        level="explicit",
        metadata=schemas.DocumentMetadata(
            message_ids=[1],
            message_created_at="2026-01-01T00:00:00Z",
        ),
    )


def _legacy_oversized_document(
    content: str, session: models.Session
) -> schemas.DocumentCreate:
    """Build an over-cap document without re-validating it.

    Models a row produced before the schema cap existed (or by any future path
    that bypasses the boundary), so the table constraint and the CRUD guard are
    still exercised.
    """
    return _document("valid", session).model_copy(update={"content": content})


async def _count_documents(db_session: AsyncSession, workspace_name: str) -> int:
    result = await db_session.execute(
        select(func.count())
        .select_from(models.Document)
        .where(models.Document.workspace_name == workspace_name)
    )
    return int(result.scalar_one())


def test_table_constraint_uses_the_shared_cap():
    """The CHECK constraint reads the same constant the schema does."""
    constraint = next(
        c
        for c in models.Document.__table__.constraints
        if getattr(c, "name", None) == "ck_documents_content_length"
    )
    assert str(constraint.sqltext) == (
        f"length(content) <= {MAX_DOCUMENT_CONTENT_LENGTH}"
    )


def test_schema_rejects_65536_at_the_api_boundary():
    """65,536 is refused before it can reach the database; 65,535 is accepted."""
    session = SimpleNamespace(name="session-under-test")

    accepted = _document("a" * MAX_DOCUMENT_CONTENT_LENGTH, session)
    assert len(accepted.content) == MAX_DOCUMENT_CONTENT_LENGTH

    with pytest.raises(ValidationError) as excinfo:
        _document("a" * OVER_CAP, session)

    assert "content" in str(excinfo.value)
    assert str(MAX_DOCUMENT_CONTENT_LENGTH) in str(excinfo.value)


@pytest.mark.asyncio
async def test_oversized_document_does_not_fail_its_siblings(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """One over-cap row must not discard the valid rows in the same batch."""
    test_workspace, test_peer = sample_data
    observed, session = await _setup_scope(db_session, test_workspace, test_peer)

    valid = [_document(f"valid observation {i}", session) for i in range(3)]
    oversized = _legacy_oversized_document("x" * OVER_CAP, session)

    result = await crud.create_documents(
        db_session,
        documents=[valid[0], oversized, valid[1], valid[2]],
        workspace_name=test_workspace.name,
        observer=test_peer.name,
        observed=observed.name,
    )

    assert [d.content for d in result.created_documents] == [d.content for d in valid]
    assert result.rejected_count == 1
    assert result.rejected_documents[0].constraint == "ck_documents_content_length"
    assert result.rejected_documents[0].content_length == OVER_CAP
    assert await _count_documents(db_session, test_workspace.name) == 3


@pytest.mark.asyncio
async def test_boundary_pair_65535_persists_65536_rejects(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """The cap is inclusive: 65,535 persists, 65,536 is rejected."""
    test_workspace, test_peer = sample_data
    observed, session = await _setup_scope(db_session, test_workspace, test_peer)

    at_cap = await crud.create_documents(
        db_session,
        documents=[_document("y" * MAX_DOCUMENT_CONTENT_LENGTH, session)],
        workspace_name=test_workspace.name,
        observer=test_peer.name,
        observed=observed.name,
    )
    assert len(at_cap.created_documents) == 1
    assert at_cap.rejected_count == 0

    over_cap = await crud.create_documents(
        db_session,
        documents=[_legacy_oversized_document("z" * OVER_CAP, session)],
        workspace_name=test_workspace.name,
        observer=test_peer.name,
        observed=observed.name,
    )
    assert over_cap.created_documents == []
    assert over_cap.rejected_count == 1
    assert str(OVER_CAP) in over_cap.rejected_documents[0].reason

    assert await _count_documents(db_session, test_workspace.name) == 1


@pytest.mark.asyncio
async def test_db_constraint_is_handled_per_row_not_only_pre_checked(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Even without the pre-check, the SAVEPOINT path rejects one row only.

    Raises the Python-side ceiling so the insert actually reaches Postgres and
    the real ``ck_documents_content_length`` constraint fires: the savepoint,
    not the fast path, has to keep the siblings alive.
    """
    test_workspace, test_peer = sample_data
    observed, session = await _setup_scope(db_session, test_workspace, test_peer)

    with patch("src.crud.document.MAX_DOCUMENT_CONTENT_LENGTH", 10_000_000):
        result = await crud.create_documents(
            db_session,
            documents=[
                _document("kept before", session),
                _legacy_oversized_document("w" * OVER_CAP, session),
                _document("kept after", session),
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )

    assert [d.content for d in result.created_documents] == [
        "kept before",
        "kept after",
    ]
    assert result.rejected_count == 1
    assert result.rejected_documents[0].constraint == "ck_documents_content_length"
    assert await _count_documents(db_session, test_workspace.name) == 2
