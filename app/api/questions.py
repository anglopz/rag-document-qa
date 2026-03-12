"""Question answering endpoint."""

from flask import current_app
from flask_restx import Namespace, Resource, fields

from app.services.embedding_service import EmbeddingService
from app.services.qa_service import QAService
from app.services.retrieval_service import retrieve_similar_chunks


def create_questions_ns() -> Namespace:
    """Create the questions namespace with all routes and models."""
    ns = Namespace("questions", description="Ask questions about uploaded documents")

    question_input = ns.model(
        "QuestionInput",
        {
            "question": fields.String(
                required=True,
                description="The question to ask",
                example="What is the statute of limitations?",
            ),
            "document_id": fields.String(
                description="Optional: filter to a specific document UUID"
            ),
            "top_k": fields.Integer(
                description="Number of source chunks to retrieve (default: 5)",
                default=5,
            ),
        },
    )

    source_model = ns.model(
        "Source",
        {
            "document_id": fields.String(description="Source document UUID"),
            "document_name": fields.String(description="Source document filename"),
            "chunk_text": fields.String(description="Relevant text passage"),
            "chunk_index": fields.Integer(description="Chunk position in document"),
            "similarity_score": fields.Float(description="Cosine similarity score"),
        },
    )

    answer_model = ns.model(
        "Answer",
        {
            "question": fields.String(description="The original question"),
            "answer": fields.String(description="Extracted answer from the document"),
            "confidence": fields.Float(description="Answer confidence score"),
            "sources": fields.List(
                fields.Nested(source_model), description="Source chunks used"
            ),
        },
    )

    @ns.route("")
    class QuestionResource(Resource):
        """Question answering resource."""

        @ns.doc("ask_question")
        @ns.expect(question_input)
        @ns.marshal_with(answer_model)
        @ns.response(400, "Invalid request")
        @ns.response(404, "No matching documents found")
        def post(self):
            """Ask a question and get an answer extracted from uploaded documents.

            The pipeline:
            1. Embeds the question using the same model as document chunks
            2. Finds the most similar chunks via cosine similarity
            3. Runs extractive QA on each candidate chunk
            4. Returns the best answer with confidence and source references
            """
            data = ns.payload or {}
            question = data.get("question", "").strip()
            if not question:
                ns.abort(400, "Question cannot be empty")

            document_id = data.get("document_id")
            top_k = data.get("top_k", current_app.config["DEFAULT_TOP_K"])

            embedding_svc = EmbeddingService(current_app.config["EMBEDDING_MODEL_PATH"])
            query_embedding = embedding_svc.embed_single(question)

            results = retrieve_similar_chunks(
                query_embedding, top_k=top_k, document_id=document_id
            )

            if not results:
                ns.abort(404, "No matching documents found. Upload documents first.")

            qa_svc = QAService(current_app.config["QA_MODEL_PATH"])

            best_answer = None
            best_confidence = -1.0

            for result in results:
                qa_result = qa_svc.answer(question, result.chunk_text)
                if qa_result.confidence > best_confidence and qa_result.answer:
                    best_answer = qa_result.answer
                    best_confidence = qa_result.confidence

            sources = [
                {
                    "document_id": r.document_id,
                    "document_name": r.document_name,
                    "chunk_text": r.chunk_text,
                    "chunk_index": r.chunk_index,
                    "similarity_score": r.similarity_score,
                }
                for r in results
            ]

            return {
                "question": question,
                "answer": best_answer or "No answer found in the provided documents.",
                "confidence": round(best_confidence, 4) if best_confidence > 0 else 0.0,
                "sources": sources,
            }

    return ns
