"""Shared numeric limits.

Kept free of imports so every layer — the ORM table definition, the pydantic
schema and the CRUD guards — can reference one value instead of repeating a
literal that is free to drift.
"""

# ``documents.content`` is TEXT, capped by a CHECK constraint. The API schema
# used to allow 100,000 characters while Postgres rejected anything above
# 65,535, so payloads passed validation and then failed the insert. Both sides
# now derive from this constant; ``tests/crud/test_document_content_cap.py``
# pins the boundary at 65,535 accepted / 65,536 rejected.
MAX_DOCUMENT_CONTENT_LENGTH = 65535
