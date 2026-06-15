"""
Classical K-Means wrapper for Crop Recommendation dataset.
Uses scikit-learn KMeans under the hood.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

from .base import BaseModel

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ClassicalKMeansModel(BaseModel):
    model_type = "classical"
    model_name = "Classical K-Means"
    description = (
        "A classical K-Means clustering algorithm using Euclidean distance. "
        "Applied to the Crop Recommendation dataset with StandardScaler preprocessing. "
        "Cluster labels are mapped to crop names via majority voting."
    )

    def __init__(self):
        self.kmeans = None
        self.scaler = None
        self.cluster_to_label = {}
        self.label_classes = []
        self.feature_names = []

    def load(self):
        km_path = os.path.join(MODEL_DIR, "classical_kmeans.joblib")
        sc_path = os.path.join(MODEL_DIR, "classical_kmeans_scaler.joblib")
        cl_path = os.path.join(MODEL_DIR, "classical_kmeans_cluster_map.joblib")

        if os.path.exists(km_path):
            self.kmeans = joblib.load(km_path)
            self.scaler = joblib.load(sc_path)
            self.cluster_to_label = joblib.load(cl_path)
            self._loaded = True
            return

        # Train from scratch
        self._train_and_save()

    def _train_and_save(self):
        df = pd.read_csv(DATASET_PATH)
        df.dropna(inplace=True)
        X = df.drop("label", axis=1)
        y = df["label"]
        self.feature_names = list(X.columns)
        self.label_classes = list(y.unique())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        k = len(np.unique(y))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)

        pred_clusters = kmeans.predict(X_test)
        cluster_to_label = {}
        for cluster in np.unique(pred_clusters):
            actual_labels = y_test[pred_clusters == cluster]
            if not actual_labels.empty:
                cluster_to_label[int(cluster)] = actual_labels.value_counts().index[0]

        mapped_preds = [cluster_to_label.get(int(c), self.label_classes[0]) for c in pred_clusters]
        acc = accuracy_score(y_test, mapped_preds)

        sil_score = silhouette_score(X_scaled, kmeans.predict(X_scaled))

        self.metrics = {
            "accuracy": round(acc, 4),
            "silhouette_score": round(float(sil_score), 4),
        }

        self.kmeans = kmeans
        self.scaler = scaler
        self.cluster_to_label = cluster_to_label

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(kmeans, os.path.join(MODEL_DIR, "classical_kmeans.joblib"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "classical_kmeans_scaler.joblib"))
        joblib.dump(cluster_to_label, os.path.join(MODEL_DIR, "classical_kmeans_cluster_map.joblib"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "classical_kmeans_metrics.joblib"))
        self._loaded = True

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        # Load feature names
        df = pd.read_csv(DATASET_PATH)
        feature_names = list(df.drop("label", axis=1).columns)
        medians = df.drop("label", axis=1).median()

        row = np.array([float(input_data.get(f, medians.get(f, 0))) for f in feature_names]).reshape(1, -1)
        scaled = self.scaler.transform(row)

        def _pred():
            cluster = int(self.kmeans.predict(scaled)[0])
            label = self.cluster_to_label.get(cluster, "unknown")
            # Confidence ≈ proximity (inverse min distance to centroid)
            distances = np.linalg.norm(self.kmeans.cluster_centers_ - scaled, axis=1)
            min_d = distances[cluster]
            confidence = float(np.exp(-min_d) / np.sum(np.exp(-distances)))
            return label, confidence

        (label, confidence), elapsed = self._timed_predict(_pred)

        # Load metrics if exists
        metrics_path = os.path.join(MODEL_DIR, "classical_kmeans_metrics.joblib")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
        }
