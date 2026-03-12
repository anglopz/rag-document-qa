"""Tests for the text chunker service."""

from app.services.text_chunker import chunk_text


class TestChunkText:
    """Tests for sliding window text chunking."""

    def test_empty_text(self):
        """Empty text should return no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        """Text shorter than chunk_size should return a single chunk."""
        text = "This is a short document."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_gets_chunked(self):
        """Long text should be split into multiple chunks."""
        text = "Word " * 200  # ~1000 characters
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_exists(self):
        """Consecutive chunks should have overlapping content."""
        text = "A" * 50 + "B" * 50 + "C" * 50 + "D" * 50
        chunks = chunk_text(text, chunk_size=60, overlap=20)
        # With overlap, some characters should appear in multiple chunks
        all_text = "".join(chunks)
        assert len(all_text) > len(text)  # overlap means repeated chars

    def test_no_empty_chunks(self):
        """All returned chunks should be non-empty."""
        text = "Hello world. " * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_custom_parameters(self):
        """Custom chunk_size and overlap should be respected."""
        text = "X" * 1000
        chunks_small = chunk_text(text, chunk_size=100, overlap=10)
        chunks_large = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks_small) > len(chunks_large)

    def test_sentence_boundary_preference(self):
        """Chunks should prefer breaking at sentence boundaries."""
        text = (
            "First sentence here. Second sentence here. "
            "Third sentence here. Fourth sentence here. "
            "Fifth sentence here. Sixth sentence here."
        )
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Most chunks should end with a period (sentence boundary)
        period_endings = sum(1 for c in chunks[:-1] if c.rstrip().endswith("."))
        assert period_endings > 0
