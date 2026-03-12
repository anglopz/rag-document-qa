"""Test fixtures for the RAG Document QA service."""

import os
import tempfile

import pytest

from app import create_app


@pytest.fixture
def app(tmp_path):
    """Create a test application with a temporary database."""
    db_path = str(tmp_path / "test.db")
    app = create_app({
        "DATABASE_PATH": db_path,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "EMBEDDING_MODEL_PATH": "models/embedding",
        "QA_MODEL_PATH": "models/qa",
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 50,
        "DEFAULT_TOP_K": 5,
        "TESTING": True,
    })
    yield app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def sample_txt_file():
    """Create a temporary text file for testing."""
    content = (
        "The statute of limitations for civil cases is generally six years. "
        "This period begins from the date the cause of action accrues. "
        "In contract disputes, the limitation period may vary depending on "
        "whether the contract is written or oral. Written contracts typically "
        "have a six-year limitation period, while oral contracts may have a "
        "shorter period of three years. The court may grant extensions in "
        "exceptional circumstances, such as when the plaintiff was unaware "
        "of the breach until a later date. This is known as the discovery rule."
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_pdf_bytes():
    """Create minimal PDF bytes for testing."""
    content = b"""%PDF-1.0
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test document.) Tj ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n

trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    return content
