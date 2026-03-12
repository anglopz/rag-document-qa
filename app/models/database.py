"""SQLite database setup and CRUD operations."""

import sqlite3
import uuid

import numpy as np
from flask import Flask, g


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


def get_db() -> sqlite3.Connection:
    """Get a database connection for the current request.

    Returns:
        SQLite connection with row factory enabled.
    """
    if "db" not in g:
        from flask import current_app

        g.db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(e=None) -> None:
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app: Flask) -> None:
    """Initialize the database schema and register teardown.

    Args:
        app: Flask application instance.
    """
    app.teardown_appcontext(close_db)

    with app.app_context():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()


# --- Document CRUD ---


def create_document(
    filename: str,
    content_type: str,
    text_content: str,
    chunks: list[str],
    embeddings: np.ndarray,
) -> dict:
    """Store a document with its chunks and embeddings.

    Args:
        filename: Original filename.
        content_type: File extension (pdf, docx, txt).
        text_content: Full extracted text.
        chunks: List of text chunks.
        embeddings: numpy array of shape (num_chunks, embedding_dim).

    Returns:
        Dict with document metadata including id, filename, num_chunks.
    """
    db = get_db()
    doc_id = str(uuid.uuid4())

    db.execute(
        "INSERT INTO documents (id, filename, content_type, text_content) VALUES (?, ?, ?, ?)",
        (doc_id, filename, content_type, text_content),
    )

    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO chunks (id, document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, i, chunk_text, embedding.astype(np.float32).tobytes()),
        )

    db.commit()

    created_at = db.execute(
        "SELECT created_at FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()[0]

    return {
        "id": doc_id,
        "filename": filename,
        "content_type": content_type,
        "num_chunks": len(chunks),
        "created_at": created_at,
    }


def get_document(doc_id: str) -> dict | None:
    """Get a document by ID with chunk count.

    Returns:
        Dict with document fields and num_chunks, or None if not found.
    """
    db = get_db()
    row = db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return None

    num_chunks = db.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)
    ).fetchone()[0]

    return {**dict(row), "num_chunks": num_chunks}


def list_documents() -> list[dict]:
    """List all documents with chunk counts."""
    db = get_db()
    rows = db.execute(
        """
        SELECT d.id, d.filename, d.content_type, d.created_at,
               COUNT(c.id) as num_chunks
        FROM documents d
        LEFT JOIN chunks c ON d.id = c.document_id
        GROUP BY d.id
        ORDER BY d.created_at DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks (cascade).

    Returns:
        True if document existed and was deleted, False otherwise.
    """
    db = get_db()
    cursor = db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()
    return cursor.rowcount > 0
