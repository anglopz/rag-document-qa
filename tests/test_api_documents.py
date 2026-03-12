"""Tests for document API endpoints."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestDocumentEndpoints:
    """Tests for document upload, listing, and management."""

    def test_list_documents_empty(self, client):
        """GET /api/documents should return empty list initially."""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.get_json()
        assert data["documents"] == []

    def test_upload_no_file(self, client):
        """POST /api/documents without file should return 400."""
        response = client.post("/api/documents")
        assert response.status_code == 400

    def test_upload_unsupported_format(self, client):
        """POST /api/documents with unsupported format should return 400."""
        data = {"file": (io.BytesIO(b"data"), "test.xlsx")}
        response = client.post(
            "/api/documents", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400

    @patch("app.services.embedding_service.EmbeddingService")
    def test_upload_txt_document(self, mock_embedding_cls, client):
        """POST /api/documents with a valid TXT file should return 201."""
        mock_svc = MagicMock()
        mock_svc.embed.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_embedding_cls.return_value = mock_svc

        content = "This is a test document. " * 50
        data = {"file": (io.BytesIO(content.encode()), "test.txt")}
        response = client.post(
            "/api/documents", data=data, content_type="multipart/form-data"
        )

        assert response.status_code == 201
        result = response.get_json()
        assert result["filename"] == "test.txt"
        assert result["content_type"] == "txt"
        assert result["num_chunks"] > 0
        assert "id" in result

    @patch("app.services.embedding_service.EmbeddingService")
    def test_get_document(self, mock_embedding_cls, client):
        """GET /api/documents/{id} should return document details."""
        mock_svc = MagicMock()
        mock_svc.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedding_cls.return_value = mock_svc

        content = "Test document content for retrieval."
        data = {"file": (io.BytesIO(content.encode()), "doc.txt")}
        upload_resp = client.post(
            "/api/documents", data=data, content_type="multipart/form-data"
        )
        doc_id = upload_resp.get_json()["id"]

        response = client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        result = response.get_json()
        assert result["id"] == doc_id
        assert result["text_content"] == content

    def test_get_document_not_found(self, client):
        """GET /api/documents/{id} with invalid ID should return 404."""
        response = client.get("/api/documents/nonexistent-id")
        assert response.status_code == 404

    @patch("app.services.embedding_service.EmbeddingService")
    def test_delete_document(self, mock_embedding_cls, client):
        """DELETE /api/documents/{id} should remove the document."""
        mock_svc = MagicMock()
        mock_svc.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedding_cls.return_value = mock_svc

        content = "Document to delete."
        data = {"file": (io.BytesIO(content.encode()), "delete_me.txt")}
        upload_resp = client.post(
            "/api/documents", data=data, content_type="multipart/form-data"
        )
        doc_id = upload_resp.get_json()["id"]

        response = client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 200

        # Verify it's gone
        response = client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 404

    def test_delete_document_not_found(self, client):
        """DELETE /api/documents/{id} with invalid ID should return 404."""
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404

    @patch("app.services.embedding_service.EmbeddingService")
    def test_list_documents_after_upload(self, mock_embedding_cls, client):
        """GET /api/documents should include uploaded documents."""
        mock_svc = MagicMock()
        mock_svc.embed.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_embedding_cls.return_value = mock_svc

        data = {"file": (io.BytesIO(b"Content here."), "listed.txt")}
        client.post(
            "/api/documents", data=data, content_type="multipart/form-data"
        )

        response = client.get("/api/documents")
        assert response.status_code == 200
        docs = response.get_json()["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"] == "listed.txt"
