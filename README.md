# RAG Document QA Service

A Retrieval-Augmented Generation service for document question answering. Upload documents (PDF, DOCX, TXT), then ask natural language questions and get answers extracted directly from the source material.

Built with Flask, ONNX Runtime, and SQLite. No PyTorch at runtime.

## Architecture

```
┌──────────────────────────────────────────────┐
│                Flask API                      │
│                                              │
│  POST /api/documents      ← Upload & process │
│  POST /api/questions       ← Ask & retrieve  │
│  GET  /api/documents       ← List documents  │
│  GET  /api/documents/{id}  ← Get document    │
│  DELETE /api/documents/{id} ← Remove document│
│  GET  /api/health          ← Health check    │
│                                              │
│  Swagger UI at /docs                         │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌─────▼──────┐
│ Upload │          │  Question  │
│Pipeline│          │  Pipeline  │
└───┬────┘          └─────┬──────┘
    │                     │
    │ 1. Parse document   │ 1. Embed question (ONNX)
    │ 2. Chunk text       │ 2. Cosine similarity search
    │ 3. Embed chunks     │ 3. Retrieve top-K chunks
    │    (ONNX)           │ 4. Run extractive QA (ONNX)
    │ 4. Store in SQLite  │ 5. Return answer + source
    │                     │
    └──────────┬──────────┘
               │
         ┌─────▼─────┐
         │  SQLite3   │
         │            │
         │ documents  │
         │ chunks     │
         │ embeddings │
         └────────────┘
```

### How the RAG Pipeline Works

**Upload Pipeline:**
1. Receive a document file (PDF, DOCX, or TXT)
2. Extract text using format-specific parsers
3. Split text into overlapping chunks (500 chars, 50 char overlap) with sentence-aware breaks
4. Generate embeddings for each chunk using all-MiniLM-L6-v2 via ONNX Runtime
5. Store document metadata, chunks, and embeddings (as BLOBs) in SQLite

**Question Pipeline:**
1. Generate an embedding for the question using the same model
2. Compute cosine similarity between the question and all stored chunk embeddings
3. Retrieve the top-K most similar chunks
4. Run extractive QA (deepset/roberta-base-squad2 via ONNX) on each candidate chunk
5. Return the best answer with confidence score and source references

### Why ONNX Runtime?

ONNX Runtime is optimized for inference: faster execution, lower memory footprint, and no training-only dependencies. Models are exported to ONNX format during the Docker build using `optimum`, and the production container runs without PyTorch installed. This keeps the container lean (~1.5GB vs ~4GB+ with PyTorch) and demonstrates production-grade ML deployment.

### Why Extractive QA?

For legal research, you want exact quotes from source material, not generated text that could hallucinate. Extractive QA highlights the precise passage in the document that answers the question, making it more trustworthy for legal professionals.

## Quick Start

### Using Docker (recommended)

```bash
docker-compose up --build
```

The service will be available at `http://localhost:5000`. Swagger UI documentation is at `http://localhost:5000/docs`.

### Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Export ONNX models (requires torch + optimum, one-time setup)
pip install torch transformers optimum[exporters]
python scripts/export_models.py

# Run the service
flask --app app run --host 0.0.0.0 --port 5000
```

## API Reference

Full interactive documentation is available at `/docs` (Swagger UI) when the service is running.

### Upload a Document

```bash
curl -X POST http://localhost:5000/api/documents \
  -F "file=@document.pdf"
```

Response (201):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "content_type": "pdf",
  "num_chunks": 15,
  "created_at": "2026-03-12T10:30:00"
}
```

### Ask a Question

```bash
curl -X POST http://localhost:5000/api/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the statute of limitations?",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "top_k": 3
  }'
```

Response (200):
```json
{
  "question": "What is the statute of limitations?",
  "answer": "The statute of limitations is 6 years for...",
  "confidence": 0.87,
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_name": "document.pdf",
      "chunk_text": "...relevant passage...",
      "chunk_index": 4,
      "similarity_score": 0.92
    }
  ]
}
```

### List Documents

```bash
curl http://localhost:5000/api/documents
```

### Get Document Details

```bash
curl http://localhost:5000/api/documents/{id}
```

### Delete a Document

```bash
curl -X DELETE http://localhost:5000/api/documents/{id}
```

### Health Check

```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Supported Document Formats

| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | PyPDF2 |
| Word | `.docx` | python-docx |
| Plain Text | `.txt` | UTF-8 decode |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_document_parser.py -v
pytest tests/test_text_chunker.py -v
pytest tests/test_api_documents.py -v
pytest tests/test_api_questions.py -v
```

Tests mock the ONNX models so they run without requiring model files or GPU.

## Project Structure

```
rag-document-qa/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration
│   ├── models/
│   │   └── database.py          # SQLite setup and CRUD
│   ├── services/
│   │   ├── document_parser.py   # PDF/DOCX/TXT extraction
│   │   ├── text_chunker.py      # Sliding window chunking
│   │   ├── embedding_service.py # ONNX embedding inference
│   │   ├── qa_service.py        # ONNX extractive QA
│   │   └── retrieval_service.py # Cosine similarity search
│   └── api/
│       ├── __init__.py          # API registration + Swagger
│       ├── documents.py         # Document endpoints
│       ├── questions.py         # Question endpoint
│       └── health.py            # Health check
├── tests/                       # pytest test suite
├── scripts/
│   └── export_models.py         # ONNX model export
├── Dockerfile                   # Multi-stage build
├── docker-compose.yml
└── requirements.txt
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | Flask + flask-restx | REST API with auto-generated Swagger docs |
| Database | SQLite3 | Document and embedding storage |
| Embeddings | all-MiniLM-L6-v2 | Sentence embeddings (384 dimensions) |
| QA Model | deepset/roberta-base-squad2 | Extractive question answering |
| Inference | ONNX Runtime | Production-optimized model inference |
| Document Parsing | PyPDF2, python-docx | PDF and DOCX text extraction |
| Testing | pytest | Unit and integration tests |

## Limitations and Potential Improvements

- **Vector search at scale:** Replace SQLite + numpy with PostgreSQL + pgvector or FAISS for approximate nearest neighbor search on large document collections.
- **Async processing:** Large documents could be processed asynchronously with a task queue (Celery) to avoid blocking the API.
- **Authentication:** Add API key or JWT authentication for multi-tenant use.
- **Streaming responses:** Stream QA results for better UX on slower models.
- **Chunk strategies:** Experiment with semantic chunking or recursive splitting for better retrieval quality.
- **Caching:** Cache embeddings and model sessions to reduce latency on repeated queries.
