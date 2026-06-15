"""
Pre-trained Hybrid QNN wrapper for Breast Cancer.
Architecture: pre(6→16→6) → QuantumLayer(2,6,3) → post(1→1)
Loads weights from the user's pre-trained .pth checkpoint.
"""
import os
import json
import glob
import time
import numpy as np
import joblib
import torch
import torch.nn as nn

from .base import BaseModel

# Absolute paths to the pre-trained model artefacts
_PRETRAINED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "projectfinalhqnn", "quantaumneualn"
)

# Also use the app's own breast cancer CSV for feature-name lookups
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer.csv")

N_FEATURES_ORIGINAL = 30
N_FEATURES_PCA = 6


# ──────────────────────────────────────────────
# Model architecture (matches the saved checkpoint)
# ──────────────────────────────────────────────
class _QuantumLayer(nn.Module):
    """
    Parameterised quantum-inspired layer matching the checkpoint shape.
    q.weights: (2, 6, 3)  — 2 layers, 6 qubits, 3 rotation params each.
    """

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(2, 6, 3, dtype=torch.float32))

    def forward(self, x):
        """x: (batch, 6) → (batch, 1)"""
        outs = []
        for layer_idx in range(2):
            w = self.weights[layer_idx]  # (6, 3)
            a = x * w[:, 0].unsqueeze(0)
            b = x * w[:, 1].unsqueeze(0)
            c = x * w[:, 2].unsqueeze(0)
            temp = torch.sin(a) + torch.cos(b) + torch.tanh(c)
            outs.append(temp.mean(dim=1, keepdim=True))
        return outs[0] + outs[1]


class _HQNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )
        self.q = _QuantumLayer()
        self.post = nn.Linear(1, 1)

    def forward(self, x):
        z = self.pre(x)
        q_out = self.q(z)
        out = self.post(q_out)
        return torch.sigmoid(out).squeeze(1)


def _remap_state_dict_keys(state_dict: dict) -> dict:
    """Handle key-name differences between training and this architecture."""
    if "pre_net.0.weight" not in state_dict:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("pre_net."):
            new_key = key.replace("pre_net.", "pre.", 1)
        elif key.startswith("q_layer."):
            new_key = key.replace("q_layer.", "q.", 1)
        elif key.startswith("post_net."):
            new_key = key.replace("post_net.", "post.", 1)
        remapped[new_key] = value
    return remapped


def _resolve_path(default: str, pattern: str) -> str:
    if os.path.exists(default):
        return default
    matches = sorted(glob.glob(os.path.join(_PRETRAINED_DIR, pattern)))
    return matches[0] if matches else default


# ──────────────────────────────────────────────
# Wrapper class
# ──────────────────────────────────────────────
class PretrainedHQNNModel(BaseModel):
    """Pre-trained Hybrid QNN loaded from .pth checkpoint."""

    model_type = "quantum"
    model_name = "Hybrid QNN (Pre-trained)"
    description = (
        "A pre-trained Hybrid Quantum Neural Network for breast cancer diagnosis. "
        "Classical layers (6→16→6) feed into a 6-qubit parameterised quantum layer "
        "(2 rotation layers × 3 params per qubit), followed by a linear classifier. "
        "Uses StandardScaler + PCA. Accuracy ~96.5% (seed 42)."
    )

    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.metrics = {}
        self.device = torch.device("cpu")

    def load(self):
        model_file = _resolve_path(
            os.path.join(_PRETRAINED_DIR, "hqnn_model.pth"),
            "hqnn_model*.pth"
        )
        scaler_file = _resolve_path(
            os.path.join(_PRETRAINED_DIR, "scaler_pca_hqnn.pkl"),
            "scaler_pca_hqnn*.pkl"
        )

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"HQNN checkpoint not found: {model_file}")
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(f"Scaler/PCA not found: {scaler_file}")

        # Load scaler + PCA
        self.scaler, self.pca = joblib.load(scaler_file)

        # Build and load the model
        self.model = _HQNNModel()
        state_dict = torch.load(model_file, map_location=self.device)
        state_dict = _remap_state_dict_keys(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Load reference metrics
        metrics_file = os.path.join(_PRETRAINED_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
            ref = None
            if isinstance(all_metrics, list):
                for item in all_metrics:
                    if isinstance(item, dict) and item.get("seed") == 42:
                        ref = item
                        break
            elif isinstance(all_metrics, dict):
                ref = all_metrics
            if ref:
                self.metrics = {
                    "accuracy": round(ref.get("accuracy", 0), 4),
                    "f1": round(ref.get("f1_macro", 0), 4),
                    "auc": round(ref.get("auroc", 0), 4),
                    "mcc": round(ref.get("mcc", 0), 4),
                }

        if not self.metrics:
            self.metrics = {"accuracy": 0.9649, "f1": 0.9619, "auc": 0.9977}

        self._loaded = True
        print(f"  ✓ Loaded pre-trained HQNN — acc {self.metrics.get('accuracy', 0):.2%}")

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        import pandas as pd

        # Build feature vector — use medians for missing features
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df.columns]
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
            feature_names = list(df.columns)
            medians = df.median()
        else:
            feature_names = [f"feature_{i}" for i in range(N_FEATURES_ORIGINAL)]
            medians = {f: 0 for f in feature_names}

        row = np.array(
            [float(input_data.get(f, medians.get(f, 0))) for f in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        def _pred():
            scaled = self.scaler.transform(row)
            pca_row = self.pca.transform(scaled)  # (1, 6)
            x = torch.tensor(pca_row, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                prob = float(self.model(x).cpu().numpy()[0])
            label = "Malignant" if prob >= 0.5 else "Benign"
            confidence = prob if prob >= 0.5 else 1.0 - prob
            return label, confidence, prob

        (label, confidence, prob), elapsed = self._timed_predict(_pred)

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "probability_malignant": round(prob, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
        }
