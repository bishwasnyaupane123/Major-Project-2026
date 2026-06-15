"""
Classical k-Nearest Neighbors wrapper for Crop Recommendation.
Uses sklearn KNeighborsClassifier trained on LDA-reduced features.

Artifacts directory: backend/models/cknn_artifacts/
  - knn_model.pkl      (fitted KNeighborsClassifier)
  - scaler.pkl         (fitted StandardScaler)
  - lda.pkl            (fitted LinearDiscriminantAnalysis)
  - label_encoder.pkl  (fitted LabelEncoder)
  - norm_bounds.npy    (shape (2, n_components): [g_min, g_max])
  - cv_results.json    (cross-validation results per k)

Best CV accuracy: 97.67% (k=6).
"""

import os
import json
import numpy as np
import joblib
from pathlib import Path
from .base import BaseModel

_MODEL_DIR = Path(__file__).parent.parent / "models" / "cknn_artifacts"


class ClassicalKNNModel(BaseModel):
    """BaseModel wrapper around Classical KNN for crop recommendation."""

    model_type = "classical"
    model_name = "Classical KNN"
    description = (
        "k-Nearest Neighbors with LDA-reduced features. "
        "StandardScaler → LDA → normalise → KNN (weighted, k=6). "
        "CV accuracy 97.67%."
    )

    def __init__(self):
        self.model = None
        self.scaler = None
        self.lda = None
        self.le = None
        self.g_min = None
        self.g_max = None
        self.metrics = {}

    def load(self):
        self.model = joblib.load(_MODEL_DIR / "knn_model.pkl")
        self.scaler = joblib.load(_MODEL_DIR / "scaler.pkl")
        self.lda = joblib.load(_MODEL_DIR / "lda.pkl")
        self.le = joblib.load(_MODEL_DIR / "label_encoder.pkl")
        bounds = np.load(_MODEL_DIR / "norm_bounds.npy")
        self.g_min, self.g_max = bounds[0], bounds[1]

        # Load CV metrics
        cv_path = _MODEL_DIR / "cv_results.json"
        if cv_path.exists():
            with open(cv_path) as f:
                cv = json.load(f)
            # Find best k
            best_k = max(cv, key=lambda k: cv[k]["mean"])
            best_acc = cv[best_k]["mean"]
            self.metrics = {
                "accuracy": round(float(best_acc), 4),
                "best_k": int(best_k),
            }
        else:
            self.metrics = {"accuracy": 0.9767, "best_k": 6}

        self._loaded = True
        print(f"  ✓ Classical KNN loaded — accuracy {self.metrics['accuracy']:.2%} (k={self.metrics['best_k']})")

    def _preprocess(self, features):
        """StandardScaler → LDA → normalise to [0, 1]."""
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        X_lda = self.lda.transform(X_scaled)
        rng = np.where((self.g_max - self.g_min) == 0, 1.0, self.g_max - self.g_min)
        X_norm = (X_lda - self.g_min) / rng
        return X_norm

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        features = [float(input_data.get(f, 0)) for f in feature_names]

        def _run():
            X_norm = self._preprocess(features)
            distances, indices = self.model.kneighbors(X_norm)
            distances = distances[0]
            indices = indices[0]

            neighbour_labels_enc = self.model._y[indices]
            neighbour_labels = self.le.inverse_transform(neighbour_labels_enc)

            # Weighted voting (inverse distance)
            weights = 1.0 / (distances + 1e-9)
            vote = {}
            for lbl, w in zip(neighbour_labels, weights):
                vote[str(lbl)] = vote.get(str(lbl), 0.0) + float(w)

            predicted_class = max(vote, key=vote.get)
            total_weight = sum(vote.values())
            confidence = vote[predicted_class] / total_weight if total_weight > 0 else 0

            return predicted_class, confidence, vote, distances.tolist(), [str(l) for l in neighbour_labels]

        (pred, conf, vote, dists, neighbours), elapsed = self._timed_predict(_run)

        return {
            "prediction": pred,
            "confidence": round(conf, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "extras": {
                "probabilities": {k: round(v / sum(vote.values()), 4) for k, v in vote.items()},
                "neighbour_distances": dists,
                "neighbour_labels": neighbours,
                "k": self.model.n_neighbors,
            },
        }
