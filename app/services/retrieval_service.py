"""Vector similarity search for document chunk retrieval."""

from dataclasses import dataclass

import numpy as np

from app.models.database import get_db


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""

    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    chunk_text: str
    similarity_score: float


def retrieve_similar_chunks(
    query_embedding: np.ndarray,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[RetrievalResult]:
    """Find the most similar document chunks to a query embedding.

    Computes cosine similarity between the query embedding and all stored
    chunk embeddings, returning the top-K most similar chunks.

    Args:
        query_embedding: 1D numpy array of the query embedding vector.
        top_k: Number of results to return.
        document_id: Optional filter to search within a specific document.

    Returns:
        List of RetrievalResult sorted by descending similarity score.
    """
    db = get_db()

    if document_id:
        rows = db.execute(
            """
            SELECT c.id, c.document_id, d.filename, c.chunk_index,
                   c.text, c.embedding
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = ?
            """,
            (document_id,),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT c.id, c.document_id, d.filename, c.chunk_index,
                   c.text, c.embedding
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            """
        ).fetchall()

    if not rows:
        return []

    results = []
    for row in rows:
        chunk_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        similarity = float(np.dot(query_embedding, chunk_embedding))
        results.append(
            RetrievalResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                document_name=row["filename"],
                chunk_index=row["chunk_index"],
                chunk_text=row["text"],
                similarity_score=round(similarity, 4),
            )
        )

    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results[:top_k]
