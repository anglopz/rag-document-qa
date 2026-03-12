"""Document upload and management endpoints."""

import werkzeug
from flask import request
from flask_restx import Namespace, Resource, fields

from app.models.database import (
    create_document,
    delete_document,
    get_document,
    list_documents,
)
from app.services.document_parser import parse_document
from app.services.text_chunker import chunk_text

ns = Namespace("documents", description="Document upload and management")

# Response models for Swagger documentation
document_model = ns.model(
    "Document",
    {
        "id": fields.String(description="Document UUID"),
        "filename": fields.String(description="Original filename"),
        "content_type": fields.String(description="File type (pdf, docx, txt)"),
        "num_chunks": fields.Integer(description="Number of text chunks"),
        "created_at": fields.String(description="Creation timestamp"),
    },
)

document_detail_model = ns.inherit(
    "DocumentDetail",
    document_model,
    {
        "text_content": fields.String(description="Full extracted text"),
    },
)

document_list_model = ns.model(
    "DocumentList",
    {
        "documents": fields.List(fields.Nested(document_model)),
    },
)

upload_parser = ns.parser()
upload_parser.add_argument(
    "file",
    location="files",
    type=werkzeug.datastructures.FileStorage,
    required=True,
    help="Document file (PDF, DOCX, or TXT)",
)


@ns.route("")
class DocumentList(Resource):
    """Document collection resource."""

    @ns.doc("list_documents")
    @ns.marshal_with(document_list_model)
    def get(self):
        """List all uploaded documents."""
        return {"documents": list_documents()}

    @ns.doc("upload_document")
    @ns.expect(upload_parser)
    @ns.marshal_with(document_model, code=201)
    @ns.response(400, "Invalid file or unsupported format")
    def post(self):
        """Upload and process a document.

        Extracts text, splits into chunks, generates embeddings,
        and stores everything in the database.
        """
        if "file" not in request.files:
            ns.abort(400, "No file provided")

        file = request.files["file"]
        if not file.filename:
            ns.abort(400, "No filename provided")

        try:
            text, content_type = parse_document(file)
        except ValueError as e:
            ns.abort(400, str(e))

        from flask import current_app

        chunks = chunk_text(
            text,
            chunk_size=current_app.config["CHUNK_SIZE"],
            overlap=current_app.config["CHUNK_OVERLAP"],
        )

        if not chunks:
            ns.abort(400, "Document produced no text chunks")

        from app.services.embedding_service import EmbeddingService

        embedding_svc = EmbeddingService(current_app.config["EMBEDDING_MODEL_PATH"])
        embeddings = embedding_svc.embed(chunks)

        doc = create_document(file.filename, content_type, text, chunks, embeddings)
        return doc, 201


@ns.route("/<string:doc_id>")
@ns.param("doc_id", "Document UUID")
class DocumentResource(Resource):
    """Single document resource."""

    @ns.doc("get_document")
    @ns.marshal_with(document_detail_model)
    @ns.response(404, "Document not found")
    def get(self, doc_id):
        """Get a document by ID with full text content."""
        doc = get_document(doc_id)
        if doc is None:
            ns.abort(404, "Document not found")
        return doc

    @ns.doc("delete_document")
    @ns.response(200, "Document deleted")
    @ns.response(404, "Document not found")
    def delete(self, doc_id):
        """Delete a document and all its chunks."""
        if not delete_document(doc_id):
            ns.abort(404, "Document not found")
        return {"message": "Document deleted"}
