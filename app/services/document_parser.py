"""Document text extraction for PDF, DOCX, and TXT files."""

import io

from PyPDF2 import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt"}


def parse_document(file_storage) -> tuple[str, str]:
    """Extract text content from an uploaded file.

    Args:
        file_storage: Werkzeug FileStorage object from the request.

    Returns:
        Tuple of (extracted_text, content_type).

    Raises:
        ValueError: If the file type is not supported or extraction fails.
    """
    filename = file_storage.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: .{extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_bytes = file_storage.read()

    if extension == "pdf":
        text = _extract_pdf(file_bytes)
    elif extension == "docx":
        text = _extract_docx(file_bytes)
    else:
        text = _extract_txt(file_bytes)

    if not text.strip():
        raise ValueError("No text content could be extracted from the document.")

    return text, extension


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _extract_txt(file_bytes: bytes) -> str:
    """Extract text from plain text bytes."""
    return file_bytes.decode("utf-8", errors="replace")
