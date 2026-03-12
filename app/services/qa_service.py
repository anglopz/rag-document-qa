"""ONNX-based extractive question answering using optimum/roberta-base-squad2."""

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.services.embedding_service import _find_onnx_model


@dataclass
class QAResult:
    """Result from extractive question answering."""

    answer: str
    confidence: float
    start: int
    end: int


class QAService:
    """Extractive question answering via ONNX Runtime.

    Uses deepset/roberta-base-squad2 exported to ONNX format. Extracts
    answer spans directly from context passages — no text generation,
    ensuring answers are exact quotes from the source material.
    """

    def __init__(self, model_path: str):
        """Initialize the QA service.

        Args:
            model_path: Path to the directory containing the ONNX model
                        and tokenizer files.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = ort.InferenceSession(
            _find_onnx_model(model_path),
            providers=["CPUExecutionProvider"],
        )

    def answer(self, question: str, context: str) -> QAResult:
        """Extract an answer span from the context for the given question.

        Args:
            question: The question to answer.
            context: The text passage to search for the answer.

        Returns:
            QAResult with the extracted answer, confidence score,
            and character offsets within the context.
        """
        encoded = self.tokenizer(
            question,
            context,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop("offset_mapping")[0]

        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        outputs = self.session.run(None, inputs)
        start_logits = outputs[0][0]
        end_logits = outputs[1][0]

        # Find the context token range (skip question tokens)
        input_ids = encoded["input_ids"][0]
        sep_indices = np.where(input_ids == self.tokenizer.sep_token_id)[0]

        if len(sep_indices) >= 2:
            # For RoBERTa: </s></s> separates question from context
            context_start_token = int(sep_indices[1]) + 1
        elif len(sep_indices) == 1:
            context_start_token = int(sep_indices[0]) + 1
        else:
            context_start_token = 1

        context_end_token = len(input_ids) - 1

        # Mask logits outside context range
        start_logits[:context_start_token] = -float("inf")
        start_logits[context_end_token:] = -float("inf")
        end_logits[:context_start_token] = -float("inf")
        end_logits[context_end_token:] = -float("inf")

        start_idx = int(np.argmax(start_logits))
        end_idx = int(np.argmax(end_logits))

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        # Calculate confidence from softmax probabilities
        start_probs = _softmax(start_logits[context_start_token:context_end_token])
        end_probs = _softmax(end_logits[context_start_token:context_end_token])
        confidence = float(
            start_probs[start_idx - context_start_token]
            * end_probs[end_idx - context_start_token]
        )

        # Map token indices back to character offsets
        start_char = int(offset_mapping[start_idx][0])
        end_char = int(offset_mapping[end_idx][1])
        answer_text = context[start_char:end_char].strip()

        return QAResult(
            answer=answer_text,
            confidence=round(confidence, 4),
            start=start_char,
            end=end_char,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()
