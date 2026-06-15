"""
Hybrid Quantum Neural Network (HQNN) for Breast Cancer Wisconsin dataset.
Architecture: Classical pre-network (30→16→6, Tanh) + Quantum StronglyEntanglingLayers + Classical post (1→1)
"""
import numpy as np
import pandas as pd
import joblib
import os
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from .base import BaseModel

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

N_QUBITS = 6
N_LAYERS = 2


def _build_qnode():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return qnode


class HQNNModel_Net(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        qnode = _build_qnode()
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.pre = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, N_QUBITS),
            nn.Tanh(),
        )
        self.q = qlayer
        self.post = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.q(x).unsqueeze(-1)
        x = self.post(x)
        return x.squeeze(-1)

    def get_pre_weights(self):
        """Return first layer weight norms as feature importance proxy."""
        w = self.pre[0].weight.detach().cpu().numpy()  # (16, in_features)
        importance = np.linalg.norm(w, axis=0)  # per feature
        importance = importance / importance.max()
        return importance.tolist()


class HQNNModelWrapper(BaseModel):
    model_type = "quantum"
    model_name = "Hybrid QNN (HQNN)"
    description = (
        "A Hybrid Quantum-Classical Neural Network. Classical layers pre-process the "
        "30 breast cancer features into 6 latent dimensions, which are then encoded "
        "into a 6-qubit circuit using AngleEmbedding. StronglyEntanglingLayers perform "
        "parameterized quantum rotations before a final classical linear layer outputs the diagnosis."
    )

    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.metrics = {}
        self.device = torch.device("cpu")
        self.n_features = N_QUBITS  # after PCA

    def load(self):
        model_path = os.path.join(MODEL_DIR, "hqnn.pt")
        if os.path.exists(model_path):
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "hqnn_scaler.joblib"))
            self.pca    = joblib.load(os.path.join(MODEL_DIR, "hqnn_pca.joblib"))
            self.model = HQNNModel_Net(N_QUBITS).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            metrics_path = os.path.join(MODEL_DIR, "hqnn_metrics.joblib")
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
            self._loaded = True
            return
        self._train_and_save()

    def _train_and_save(self):
        df = pd.read_csv(DATASET_PATH)
        drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        df.dropna(inplace=True)

        X = df.drop("diagnosis", axis=1).astype(float).values
        y = df["diagnosis"].astype(int).values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        pca = PCA(n_components=N_QUBITS).fit(X_train_s)
        X_train_p = pca.transform(X_train_s)
        X_val_p   = pca.transform(X_val_s)
        X_test_p  = pca.transform(X_test_s)

        model = HQNNModel_Net(N_QUBITS).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_t = torch.tensor(X_train_p, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        batch_size = 32
        for epoch in range(60):
            model.train()
            idx = np.random.permutation(len(X_t))
            for i in range(0, len(X_t), batch_size):
                j = idx[i:i + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_t[j]), y_t[j])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_p, dtype=torch.float32)
            probs = torch.sigmoid(model(X_test_t)).numpy()
            preds = (probs >= 0.5).astype(int)

        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "auc": round(float(roc_auc_score(y_test, probs)), 4),
            "f1": round(float(f1_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
        }

        self.model = model
        self.scaler = scaler
        self.pca = pca

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hqnn.pt"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "hqnn_scaler.joblib"))
        joblib.dump(pca, os.path.join(MODEL_DIR, "hqnn_pca.joblib"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "hqnn_metrics.joblib"))
        self._loaded = True

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        df = pd.read_csv(DATASET_PATH)
        drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
        df.drop(columns=drop_cols + ["diagnosis"], inplace=True, errors="ignore")
        feature_names = list(df.columns)
        medians = df.median()

        row = np.array([float(input_data.get(f, medians.get(f, 0))) for f in feature_names]).reshape(1, -1)
        scaled = self.scaler.transform(row)
        pca_row = self.pca.transform(scaled)

        def _pred():
            t = torch.tensor(pca_row, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logit = self.model(t).item()
                prob = float(torch.sigmoid(torch.tensor(logit)))
            label = "Malignant" if prob >= 0.5 else "Benign"
            confidence = prob if prob >= 0.5 else 1 - prob
            return label, confidence, prob

        (label, confidence, prob), elapsed = self._timed_predict(_pred)

        # Feature importance from classical pre-layer
        feature_importance = None
        try:
            raw_importance = self.model.get_pre_weights()
            # Map PCA components back — approximate per original feature via PCA loadings
            pca_components = self.pca.components_  # (N_QUBITS, 30)
            imp = np.zeros(len(feature_names))
            for qi, weight in enumerate(raw_importance[:N_QUBITS]):
                imp += abs(weight) * np.abs(pca_components[qi])
            imp /= imp.max()
            feature_importance = {f: round(float(v), 3) for f, v in zip(feature_names, imp)}
        except Exception:
            pass

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "probability_malignant": round(prob, 4),
            "inference_time_ms": elapsed,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "feature_importance": feature_importance,
            "circuit_info": {
                "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "circuit_type": "StronglyEntanglingLayers",
                "description": "AngleEmbedding (RY) + StronglyEntanglingLayers"
            }
        }
