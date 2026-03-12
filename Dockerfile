# Stage 1: Download pre-exported ONNX models from HuggingFace Hub
FROM python:3.11-slim AS model-builder
WORKDIR /models
RUN pip install --no-cache-dir huggingface-hub

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'sentence-transformers/all-MiniLM-L6-v2', \
    local_dir='/models/embedding', \
    allow_patterns=['onnx/*', 'tokenizer*', 'special_tokens*', 'vocab*', 'config.json', 'sentence*'], \
)"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'optimum/roberta-base-squad2', \
    local_dir='/models/qa', \
    allow_patterns=['model.onnx', 'tokenizer*', 'special_tokens*', 'vocab*', 'config.json', 'merges*'], \
)"


# Stage 2: Lean runtime — no PyTorch, only ONNX Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-exported ONNX models from builder stage
COPY --from=model-builder /models /app/models

# Copy application code
COPY app/ app/
COPY tests/ tests/

# Create data directory for SQLite
RUN mkdir -p /app/data /app/uploads

EXPOSE 5000

CMD ["python", "-m", "flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5000"]
