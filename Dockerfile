# Stage 1: Export HuggingFace models to ONNX format
# This stage requires PyTorch and transformers — heavy but temporary
FROM python:3.11-slim AS model-builder

WORKDIR /export

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    optimum[exporters]==1.24.0 \
    onnxruntime==1.21.0

COPY scripts/export_models.py .
RUN python export_models.py --output-dir /models


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

# Use gunicorn for production
CMD ["python", "-m", "flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5000"]
