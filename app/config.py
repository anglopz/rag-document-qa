"""Application configuration."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config:
    """Base configuration."""

    DATABASE_PATH = os.environ.get(
        "DATABASE_PATH", os.path.join(BASE_DIR, "data", "rag.db")
    )
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads")
    )
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

    # Model paths
    EMBEDDING_MODEL_PATH = os.environ.get(
        "EMBEDDING_MODEL_PATH", os.path.join(BASE_DIR, "models", "embedding")
    )
    QA_MODEL_PATH = os.environ.get(
        "QA_MODEL_PATH", os.path.join(BASE_DIR, "models", "qa")
    )

    # Chunking parameters
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))

    # Retrieval parameters
    DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))


class TestConfig(Config):
    """Test configuration."""

    TESTING = True
    DATABASE_PATH = ":memory:"
