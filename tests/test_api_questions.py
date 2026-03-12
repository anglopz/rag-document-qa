"""Tests for the question answering endpoint."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.services.qa_service import QAResult


class TestQuestionEndpoint:
    """Tests for POST /api/questions."""

    @patch("app.api.questions.EmbeddingService")
    def test_question_no_documents(self, mock_emb_cls, client):
        """Asking a question with no documents should return 404."""
        mock_svc = MagicMock()
        mock_svc.embed_single.return_value = np.random.rand(384).astype(np.float32)
        mock_emb_cls.return_value = mock_svc

        response = client.post(
            "/api/questions",
            json={"question": "What is the answer?"},
        )
        assert response.status_code == 404

    def test_question_empty_string(self, client):
        """Empty question should return 400."""
        response = client.post(
            "/api/questions",
            json={"question": ""},
        )
        assert response.status_code == 400

    def test_question_missing_field(self, client):
        """Missing question field should return 400."""
        response = client.post(
            "/api/questions",
            json={},
        )
        assert response.status_code == 400

    @patch("app.api.questions.QAService")
    @patch("app.api.questions.EmbeddingService")
    @patch("app.services.embedding_service.EmbeddingService")
    def test_question_with_document(
        self, mock_doc_emb_cls, mock_q_emb_cls, mock_qa_cls, client
    ):
        """Full pipeline: upload document, ask question, get answer."""
        embedding = np.random.rand(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Mock embedding service for document upload
        mock_doc_emb = MagicMock()
        mock_doc_emb.embed.return_value = embedding.reshape(1, -1)
        mock_doc_emb_cls.return_value = mock_doc_emb

        # Mock embedding service for question
        mock_q_emb = MagicMock()
        mock_q_emb.embed_single.return_value = embedding
        mock_q_emb_cls.return_value = mock_q_emb

        # Mock QA service
        mock_qa = MagicMock()
        mock_qa.answer.return_value = QAResult(
            answer="six years",
            confidence=0.85,
            start=10,
            end=20,
        )
        mock_qa_cls.return_value = mock_qa

        # Upload a document
        data = {
            "file": (
                io.BytesIO(b"The statute of limitations is six years."),
                "legal.txt",
            )
        }
        upload_resp = client.post(
            "/api/documents",
            data=data,
            content_type="multipart/form-data",
        )
        assert upload_resp.status_code == 201
        doc_id = upload_resp.get_json()["id"]

        # Ask a question
        response = client.post(
            "/api/questions",
            json={
                "question": "What is the statute of limitations?",
                "document_id": doc_id,
            },
        )
        assert response.status_code == 200
        result = response.get_json()
        assert result["answer"] == "six years"
        assert result["confidence"] > 0
        assert len(result["sources"]) > 0
        assert result["sources"][0]["document_name"] == "legal.txt"
