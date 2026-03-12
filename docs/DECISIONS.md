# Architecture Decision Records

## ADR-006: Download pre-exported ONNX models instead of exporting at build time

### Status

Accepted

### Context

The original design called for exporting HuggingFace models to ONNX format during
the Docker build using `optimum-cli export onnx`. This required PyTorch, transformers,
optimum, and onnxscript in the builder stage.

In practice, the ONNX export toolchain has version incompatibilities between:
- PyTorch's dynamo-based ONNX exporter (torch >= 2.5) requiring onnxscript
- optimum's opset version requirements conflicting with newer torch versions
- torch 2.4.x working with the legacy exporter but having its own optimum constraints

These dependencies create a fragile build that breaks across version combinations.

### Decision

Download pre-exported ONNX models directly from HuggingFace Hub using
`huggingface-cli download`. `sentence-transformers/all-MiniLM-L6-v2` has pre-exported ONNX files on the Hub.
`deepset/roberta-base-squad2` does not ship ONNX exports, so we use
`optimum/roberta-base-squad2` — the same model weights pre-exported by the
HuggingFace Optimum team.

The Docker build remains multi-stage:
- **Stage 1:** Downloads ONNX model files and tokenizer configs (only needs
  `huggingface-hub`, ~10MB vs ~2GB for PyTorch)
- **Stage 2:** Lean runtime with only `onnxruntime` — no PyTorch, no optimum

### Rationale

1. **Production robustness:** Pre-exported models from the Hub are immutable,
   validated artifacts. Exporting at build time couples your CI/CD to the fragile
   intersection of PyTorch, optimum, and onnxscript version compatibility.

2. **Build efficiency:** PyTorch alone is ~2GB. Exporting adds 5-10 minutes and
   significant RAM to every build. Downloading pre-exported ONNX files (~100MB)
   keeps Stage 1 lightweight and fast.

3. **Deterministic CI/CD:** The ONNX model is treated as a static asset, not a
   build-time transformation. This is the same pattern used for any compiled
   artifact — you don't recompile your dependencies from source in production.

### Consequences

**Positive:**
- Deterministic builds — no export step that can fail due to version drift
- Faster builds — downloading ~100MB of model files vs installing PyTorch + exporting
- Smaller builder stage — only `huggingface-hub` needed
- Same production result — Stage 2 is identical either way

**Negative:**
- Dependent on HuggingFace Hub availability at build time
- Models are whatever version the Hub has, not pinned to a specific export config

### Production alternative

In a CI/CD pipeline, models would be exported via `optimum-cli` on a dedicated
machine with tested, pinned dependency versions, then cached as build artifacts
(e.g., in S3 or a container registry layer). The Dockerfile would `COPY` from a
local path or download from the artifact store, avoiding both the version
fragility and the Hub dependency.

For custom or fine-tuned models that don't have pre-exported ONNX versions on the
Hub, the export would be run as a separate CI/CD step on a machine with pinned,
tested dependency versions. The resulting `.onnx` file would be cached as a build
artifact and `COPY`'d into the runtime image.
