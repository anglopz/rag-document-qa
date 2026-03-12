"""API registration and Swagger configuration."""

from flask import Flask
from flask_restx import Api


api = Api(
    title="RAG Document QA Service",
    version="1.0",
    description=(
        "A Retrieval-Augmented Generation service for document question answering. "
        "Upload documents (PDF, DOCX, TXT), then ask questions and get answers "
        "extracted directly from the source material using ONNX-optimized models."
    ),
    doc="/docs",
)


def register_api(app: Flask) -> None:
    """Register the API with namespaces on the Flask app."""
    from app.api.documents import ns as documents_ns
    from app.api.questions import ns as questions_ns
    from app.api.health import ns as health_ns

    api.add_namespace(documents_ns, path="/api/documents")
    api.add_namespace(questions_ns, path="/api/questions")
    api.add_namespace(health_ns, path="/api/health")

    api.init_app(app)
