"""Text chunking with sliding window for RAG pipeline."""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks using a sliding window.

    Uses character-based chunking with overlap to preserve context
    across chunk boundaries. Breaks are placed at sentence boundaries
    when possible to avoid cutting mid-sentence.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks. Empty list if input text is empty.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for the last sentence-ending punctuation within the chunk
            break_point = _find_break_point(text, start, end)
            if break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk length - overlap)
        step = max(end - start - overlap, 1)
        start += step

    return chunks


def _find_break_point(text: str, start: int, end: int) -> int:
    """Find the best position to break text, preferring sentence boundaries.

    Searches backwards from `end` for sentence-ending punctuation (. ! ?),
    falling back to the original end position if none found within a
    reasonable range (last 20% of the chunk).

    Args:
        text: The full text being chunked.
        start: Start index of the current chunk.
        end: Proposed end index of the current chunk.

    Returns:
        The adjusted end position for the chunk.
    """
    search_start = start + int((end - start) * 0.8)

    for i in range(end - 1, search_start - 1, -1):
        if text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1] in " \n\t"):
            return i + 1

    return end
