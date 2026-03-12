"""Export HuggingFace models to ONNX format.

This script is run during Docker build to convert the embedding and QA models
to ONNX format. The resulting ONNX files are used at runtime with onnxruntime,
eliminating the need for PyTorch in production.

Usage:
    python scripts/export_models.py [--output-dir models]
"""

import argparse
import os

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForQuestionAnswering
from transformers import AutoTokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"


def export_embedding_model(output_dir: str) -> None:
    """Export the sentence-transformer embedding model to ONNX."""
    path = os.path.join(output_dir, "embedding")
    print(f"Exporting embedding model ({EMBEDDING_MODEL}) to {path}...")

    model = ORTModelForFeatureExtraction.from_pretrained(EMBEDDING_MODEL, export=True)
    model.save_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    tokenizer.save_pretrained(path)

    print("Embedding model exported successfully.")


def export_qa_model(output_dir: str) -> None:
    """Export the extractive QA model to ONNX."""
    path = os.path.join(output_dir, "qa")
    print(f"Exporting QA model ({QA_MODEL}) to {path}...")

    model = ORTModelForQuestionAnswering.from_pretrained(QA_MODEL, export=True)
    model.save_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    tokenizer.save_pretrained(path)

    print("QA model exported successfully.")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save exported models (default: models)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    export_embedding_model(args.output_dir)
    export_qa_model(args.output_dir)

    print(f"\nAll models exported to {args.output_dir}/")


if __name__ == "__main__":
    main()
