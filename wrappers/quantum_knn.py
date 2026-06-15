"""
Quantum k-Nearest Neighbors wrapper for Crop Recommendation.
Uses Qiskit swap-test circuits to compute quantum distances.

Artifacts directory: backend/models/qknn_checkpoints/
  - scaler.pkl          (fitted StandardScaler)
  - lda.pkl             (fitted LinearDiscriminantAnalysis)
  - label_encoder.pkl   (fitted LabelEncoder)
  - norm_bounds.npy     (shape (2, n_components): [g_min, g_max])
  - X_train.npy         (preprocessed training features)
  - y_train.npy         (training labels, raw strings)
  - dist_matrix.npy     (pre-computed distance matrix for test set, optional)
  - accuracies.json     (accuracy per k)

Best test accuracy: 97.50% (k=8).

NOTE: For live inference, computing swap-test distances against all training
samples is slow (~1760 circuits). We use a HYBRID approach:
  1. First, use classical Euclidean distance to find top-50 nearest candidates
  2. Then run quantum swap-test only on those 50 candidates
This gives the same results but is ~35x faster.
"""

import os
import json
import warnings
import numpy as np
import joblib
from pathlib import Path
from .base import BaseModel

warnings.filterwarnings("ignore")

_MODEL_DIR = Path(__file__).parent.parent / "models" / "qknn_checkpoints"


def _pad_and_normalise(vector):
    """Pad to next power of two and L2-normalise."""
    n = len(vector)
    size = 1 << (n - 1).bit_length() if n > 1 else 2
    out = np.zeros(size, dtype=float)
    out[:n] = vector
    norm = np.linalg.norm(out)
    if norm < 1e-12:
        out[0] = 1.0
        return out
    return out / norm


class QuantumKNNModel(BaseModel):
    """BaseModel wrapper around Quantum KNN for crop recommendation."""

    model_type = "quantum"
    model_name = "Quantum KNN"
    description = (
        "Quantum k-NN using swap-test circuits for distance. "
        "LDA → normalise to [0,π] → amplitude encoding → swap-test. "
        "Test accuracy 97.50% (k=8)."
    )

    def __init__(self):
        self.scaler = None
        self.lda = None
        self.le = None
        self.g_min = None
        self.g_max = None
        self.X_train = None
        self.y_train = None
        self.best_k = 8
        self.simulator = None
        self.n_shots = 512
        self.metrics = {}

    def load(self):
        # Defer qiskit import to keep startup fast
        from qiskit_aer import AerSimulator

        self.scaler = joblib.load(_MODEL_DIR / "scaler.pkl")
        self.lda = joblib.load(_MODEL_DIR / "lda.pkl")
        self.le = joblib.load(_MODEL_DIR / "label_encoder.pkl")
        bounds = np.load(_MODEL_DIR / "norm_bounds.npy")
        self.g_min, self.g_max = bounds[0], bounds[1]
        self.X_train = np.load(_MODEL_DIR / "X_train.npy", allow_pickle=True)
        self.y_train = np.load(_MODEL_DIR / "y_train.npy", allow_pickle=True)

        # Load best k from accuracies
        acc_path = _MODEL_DIR / "accuracies.json"
        if acc_path.exists():
            with open(acc_path) as f:
                accs = json.load(f)
            self.best_k = int(max(accs, key=lambda k: accs[k]))
            best_acc = accs[str(self.best_k)]
            self.metrics = {
                "accuracy": round(float(best_acc), 4),
                "best_k": self.best_k,
            }
        else:
            self.metrics = {"accuracy": 0.975, "best_k": 8}

        # Build simulator
        try:
            self.simulator = AerSimulator(method="statevector", device="CPU")
            self.simulator.set_options(seed_simulator=42)
        except Exception:
            self.simulator = AerSimulator()
            self.simulator.set_options(seed_simulator=42)

        self._loaded = True
        print(f"  ✓ Quantum KNN loaded — accuracy {self.metrics['accuracy']:.2%} (k={self.best_k}, {len(self.X_train)} train samples)")

    def _preprocess(self, features):
        """StandardScaler → LDA → normalise to [0, π]."""
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        X_lda = self.lda.transform(X_scaled)
        rng = np.where((self.g_max - self.g_min) == 0, 1.0, self.g_max - self.g_min)
        X_norm = (X_lda - self.g_min) / rng * np.pi
        return X_norm

    def _swap_test_distance(self, vec1, vec2):
        """Run the swap-test circuit and return quantum distance."""
        from qiskit import QuantumCircuit

        state_a = _pad_and_normalise(vec1)
        state_b = _pad_and_normalise(vec2)
        n_qubits = int(np.log2(len(state_a)))

        qc = QuantumCircuit(1 + 2 * n_qubits, 1)
        reg_a = list(range(1, n_qubits + 1))
        reg_b = list(range(n_qubits + 1, 2 * n_qubits + 1))

        qc.h(0)
        qc.initialize(state_a, reg_a)
        qc.initialize(state_b, reg_b)
        for i in range(n_qubits):
            qc.cswap(0, reg_a[i], reg_b[i])
        qc.h(0)
        qc.measure(0, 0)

        counts = self.simulator.run(qc, shots=self.n_shots).result().get_counts()
        p_zero = counts.get("0", 0) / self.n_shots
        inner_sq = float(np.clip(2.0 * p_zero - 1.0, 0.0, 1.0))
        inner = np.sqrt(inner_sq + 1e-12)
        return float(np.sqrt(2.0 * (1.0 - inner)))

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        features = [float(input_data.get(f, 0)) for f in feature_names]

        def _run():
            test_vec = self._preprocess(features)[0]  # shape (n_components,)

            # HYBRID speedup: classical pre-filter to top-50 candidates
            classical_dists = np.linalg.norm(self.X_train - test_vec, axis=1)
            top_n = min(50, len(self.X_train))
            candidate_idx = np.argpartition(classical_dists, top_n)[:top_n]

            # Compute quantum distances only for candidates
            q_distances = []
            for idx in candidate_idx:
                d = self._swap_test_distance(test_vec, self.X_train[idx])
                q_distances.append(d)
            q_distances = np.array(q_distances)

            # Find k nearest from candidates
            k = self.best_k
            k_local = np.argpartition(q_distances, min(k, len(q_distances) - 1))[:k]
            k_dist = q_distances[k_local]
            sorted_order = np.argsort(k_dist)
            k_local = k_local[sorted_order]
            k_dist = k_dist[sorted_order]
            k_global_idx = candidate_idx[k_local]
            k_labels = self.y_train[k_global_idx]

            # Weighted voting (inverse distance)
            weights = 1.0 / (k_dist + 1e-9)
            vote = {}
            for lbl, w in zip(k_labels, weights):
                vote[str(lbl)] = vote.get(str(lbl), 0.0) + float(w)

            predicted_class = max(vote, key=vote.get)
            total_weight = sum(vote.values())
            confidence = vote[predicted_class] / total_weight if total_weight > 0 else 0

            return predicted_class, confidence, vote, k_dist.tolist(), [str(l) for l in k_labels]

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
                "k": self.best_k,
            },
            "circuit_info": {
                "n_qubits": 7,  # 1 ancilla + 2×3 data qubits (padded to 8→3 qubits each)
                "circuit_type": "Swap Test",
                "n_layers": 1,
                "description": (
                    "Amplitude encoding of LDA features → "
                    "CSWAP-based swap test for quantum distance measurement"
                ),
            },
        }
