"""
Pre-trained Classical Transformer wrapper for IMDB Sentiment Analysis.
Architecture: DistilBERT fine-tuned for sequence classification (2 classes).
Loads from the user's saved HuggingFace model directory.
"""
import os
import json
import time
import numpy as np

from .base import BaseModel

# Absolute path to the saved HuggingFace model directory
_PRETRAINED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "projectfinalhqnn", "classicalsentimentanlaysis", "sentiment_model"
)
METRICS_FILE = os.path.join(_PRETRAINED_DIR, "metrics.json")


class PretrainedClassicalTransformerModel(BaseModel):
    """Fine-tuned DistilBERT loaded from the user's pre-trained checkpoint."""

    model_type = "classical"
    model_name = "Classical Transformer (DistilBERT)"
    description = (
        "A pre-trained DistilBERT model fine-tuned on IMDB movie reviews for "
        "binary sentiment classification. Uses HuggingFace tokenizer with "
        "max_length=128. Test accuracy ~87.6%."
    )

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.metrics = {}

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not os.path.exists(_PRETRAINED_DIR):
            raise FileNotFoundError(
                f"Pre-trained sentiment model directory not found: {_PRETRAINED_DIR}"
            )

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(_PRETRAINED_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(_PRETRAINED_DIR)
        self.model.to(self.device)
        self.model.eval()

        # Load reference metrics
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            val = raw.get("validation_results", {})
            self.metrics = {
                "accuracy": round(raw.get("test_accuracy", val.get("eval_accuracy", 0)), 4),
                "precision": round(raw.get("test_precision", 0), 4),
                "recall": round(raw.get("test_recall", 0), 4),
                "f1": round(raw.get("test_f1", 0), 4),
            }
        else:
            self.metrics = {"accuracy": 0.8757}

        self._loaded = True
        print(
            f"  ✓ Loaded pre-trained DistilBERT Sentiment — "
            f"acc {self.metrics.get('accuracy', 0):.2%}"
        )

    def predict(self, input_data: dict) -> dict:
        import torch
        self.ensure_loaded()

        text = str(input_data.get("review", "")).strip()
        if not text:
            text = "No review text provided."

        def _pred():
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            # DistilBERT does not accept token_type_ids — remove if present
            enc.pop("token_type_ids", None)

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=1)

            label_idx = int(torch.argmax(probs).item())
            label = "Positive" if label_idx == 1 else "Negative"
            confidence = float(probs[0][label_idx])
            prob_positive = float(probs[0][1])
            return label, confidence, prob_positive

        (label, confidence, prob_positive), elapsed = self._timed_predict(_pred)

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "probability_positive": round(prob_positive, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
        }
