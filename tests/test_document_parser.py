"""Tests for the document parser service."""

import io
import pytest

from werkzeug.datastructures import FileStorage

from app.services.document_parser import parse_document, SUPPORTED_EXTENSIONS


class TestParseDocument:
    """Tests for document text extraction."""

    def test_parse_txt_file(self, sample_txt_file):
        """Text files should be parsed correctly."""
        with open(sample_txt_file, "rb") as f:
            file_storage = FileStorage(f, filename="test.txt")
            text, content_type = parse_document(file_storage)

        assert content_type == "txt"
        assert "statute of limitations" in text
        assert len(text) > 0

    def test_parse_pdf_file(self, sample_pdf_bytes):
        """PDF files should be parsed correctly."""
        file_storage = FileStorage(
            io.BytesIO(sample_pdf_bytes), filename="test.pdf"
        )
        text, content_type = parse_document(file_storage)

        assert content_type == "pdf"
        assert len(text.strip()) > 0

    def test_unsupported_file_type(self):
        """Unsupported file types should raise ValueError."""
        file_storage = FileStorage(
            io.BytesIO(b"data"), filename="test.xlsx"
        )
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(file_storage)

    def test_empty_file(self):
        """Empty files should raise ValueError."""
        file_storage = FileStorage(
            io.BytesIO(b""), filename="empty.txt"
        )
        with pytest.raises(ValueError, match="No text content"):
            parse_document(file_storage)

    def test_supported_extensions(self):
        """All expected extensions should be in the supported set."""
        assert SUPPORTED_EXTENSIONS == {"pdf", "docx", "txt"}
