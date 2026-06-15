"""
Quantum K-Means wrapper using Swap-Test circuit for distance measurement.
PCA reduces to 4 dimensions for quantum compatibility.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import pennylane as qml

from .base import BaseModel

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

N_QUBITS = 4


def _build_swap_dev():
    return qml.device("default.qubit", wires=2 * N_QUBITS + 1)


_swap_dev = None


def _get_swap_test():
    global _swap_dev
    if _swap_dev is None:
        _swap_dev = _build_swap_dev()

    @qml.qnode(_swap_dev)
    def swap_test(a, b):
        qml.Hadamard(wires=0)
        for i, val in enumerate(a):
            qml.RY(float(val), wires=i + 1)
        for i, val in enumerate(b):
            qml.RY(float(val), wires=i + N_QUBITS + 1)
        for i in range(N_QUBITS):
            qml.CSWAP(wires=[0, i + 1, i + N_QUBITS + 1])
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    return swap_test


def quantum_distance(a, b):
    swap_test = _get_swap_test()
    return float(swap_test(a, b)[1])


def quantum_kmeans(X, k, iterations=5):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=int)

    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        for idx, x in enumerate(X):
            distances = [quantum_distance(x, c) for c in centroids]
            cluster_id = int(np.argmin(distances))
            clusters[cluster_id].append(idx)
            labels[idx] = cluster_id

        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(np.mean(X[clusters[i]], axis=0))
            else:
                new_centroids.append(centroids[i])
        centroids = np.array(new_centroids)

    return labels, centroids


class QuantumKMeansModel(BaseModel):
    model_type = "quantum"
    model_name = "Quantum K-Means"
    description = (
        "A hybrid quantum K-Means algorithm using a Swap-Test circuit to compute "
        "inner-product-based distances between data points encoded as quantum states. "
        "PCA reduces the crop features to 4 dimensions to fit the quantum circuit."
    )

    def __init__(self):
        self.centroids = None
        self.cluster_to_label = {}
        self.scaler = None
        self.pca = None
        self.metrics = {}

    def load(self):
        cent_path = os.path.join(MODEL_DIR, "q_kmeans_centroids.npy")
        if os.path.exists(cent_path):
            self.centroids = np.load(cent_path, allow_pickle=True)
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "q_kmeans_scaler.joblib"))
            self.pca = joblib.load(os.path.join(MODEL_DIR, "q_kmeans_pca.joblib"))
            self.cluster_to_label = joblib.load(os.path.join(MODEL_DIR, "q_kmeans_cluster_map.joblib"))
            metrics_path = os.path.join(MODEL_DIR, "q_kmeans_metrics.joblib")
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
            self._loaded = True
            return
        self._train_and_save()

    def _train_and_save(self):
        df = pd.read_csv(DATASET_PATH)
        df.dropna(inplace=True)
        X = df.drop("label", axis=1).values.astype(float)
        y = df["label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=N_QUBITS)
        X_pca = pca.fit_transform(X_scaled)

        # Use a small sample for quantum clustering (speed)
        sample_size = min(200, len(X_pca))
        idx_sample = np.random.choice(len(X_pca), sample_size, replace=False)
        X_sample = X_pca[idx_sample]
        y_sample = y.iloc[idx_sample]

        k = len(np.unique(y))
        train_labels, centroids = quantum_kmeans(X_sample, k, iterations=3)

        cluster_to_label = {}
        for c in np.unique(train_labels):
            labels_in_cluster = y_sample.iloc[train_labels == c]
            if not labels_in_cluster.empty:
                cluster_to_label[int(c)] = labels_in_cluster.value_counts().index[0]

        # Evaluate
        pred_clusters = np.array([
            int(np.argmin([quantum_distance(x, cent) for cent in centroids]))
            for x in X_sample
        ])
        mapped = [cluster_to_label.get(int(c), y.iloc[0]) for c in pred_clusters]
        acc = accuracy_score(y_sample, mapped)
        try:
            sil = silhouette_score(X_sample, train_labels)
        except Exception:
            sil = 0.0

        self.metrics = {"accuracy": round(float(acc), 4), "silhouette_score": round(float(sil), 4)}
        self.centroids = centroids
        self.scaler = scaler
        self.pca = pca
        self.cluster_to_label = cluster_to_label

        os.makedirs(MODEL_DIR, exist_ok=True)
        np.save(os.path.join(MODEL_DIR, "q_kmeans_centroids.npy"), centroids)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "q_kmeans_scaler.joblib"))
        joblib.dump(pca, os.path.join(MODEL_DIR, "q_kmeans_pca.joblib"))
        joblib.dump(cluster_to_label, os.path.join(MODEL_DIR, "q_kmeans_cluster_map.joblib"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "q_kmeans_metrics.joblib"))
        self._loaded = True

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        df = pd.read_csv(DATASET_PATH)
        feature_names = list(df.drop("label", axis=1).columns)
        medians = df.drop("label", axis=1).median()
        row = np.array([float(input_data.get(f, medians.get(f, 0))) for f in feature_names]).reshape(1, -1)
        scaled = self.scaler.transform(row)
        pca_row = self.pca.transform(scaled)[0]

        def _pred():
            distances = [quantum_distance(pca_row, c) for c in self.centroids]
            cluster = int(np.argmin(distances))
            label = self.cluster_to_label.get(cluster, "unknown")
            min_d = distances[cluster]
            neg_distances = [-d for d in distances]
            exp_vals = np.exp(neg_distances)
            confidence = float(exp_vals[cluster] / np.sum(exp_vals))
            return label, confidence

        (label, confidence), elapsed = self._timed_predict(_pred)

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "circuit_info": {
                "n_qubits": 2 * N_QUBITS + 1,
                "circuit_type": "Swap-Test",
                "description": "Hadamard + AngleEmbedding (RY) + CSWAP gates"
            }
        }
