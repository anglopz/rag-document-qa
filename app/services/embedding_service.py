"""ONNX-based embedding service using all-MiniLM-L6-v2."""

import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def _find_onnx_model(model_path: str) -> str:
    """Locate model.onnx, checking both root and onnx/ subdirectory."""
    candidates = [
        os.path.join(model_path, "model.onnx"),
        os.path.join(model_path, "onnx", "model.onnx"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No model.onnx found in {model_path} or {model_path}/onnx/"
    )


class EmbeddingService:
    """Generate text embeddings using a sentence-transformer model via ONNX Runtime.

    Uses all-MiniLM-L6-v2 exported to ONNX format for efficient CPU inference
    without PyTorch dependency.
    """

    def __init__(self, model_path: str):
        """Initialize the embedding service.

        Args:
            model_path: Path to the directory containing the ONNX model
                        and tokenizer files.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = ort.InferenceSession(
            _find_onnx_model(model_path),
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Applies mean pooling over token embeddings with attention mask
        weighting, matching the sentence-transformers pooling strategy.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim) with L2-normalized
            embeddings.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # Add token_type_ids if the model expects it
        input_names = [inp.name for inp in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = encoded.get(
                "token_type_ids",
                np.zeros_like(encoded["input_ids"]),
            ).astype(np.int64)

        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling with attention mask
        mask = encoded["attention_mask"][..., np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask, axis=1)
        counted = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = summed / counted

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text to embed.

        Returns:
            1D numpy array of the embedding vector.
        """
        return self.embed([text])[0]
