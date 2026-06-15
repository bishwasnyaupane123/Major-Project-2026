"""
Classical DNN for Breast Cancer Wisconsin dataset.
Architecture: Dense(32, relu) -> Dropout(0.25) -> Dense(16, relu) -> Dense(1, sigmoid)
Uses PyTorch (mirrors the TensorFlow model in the original notebook).
"""
import numpy as np
import pandas as pd
import joblib
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from .base import BaseModel

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "breast_cancer.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class _DNNNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ClassicalDNNModel(BaseModel):
    model_type = "classical"
    model_name = "Classical ANN"
    description = (
        "A classical Deep Neural Network (3-layer MLP) trained on the Breast Cancer "
        "Wisconsin dataset. Features are standardized. Architecture: 32→16→1 with "
        "ReLU activations and Dropout regularization."
    )

    def __init__(self):
        self.model = None
        self.scaler = None
        self.n_features = 30
        self.device = torch.device("cpu")
        self.metrics = {}

    def load(self):
        model_path = os.path.join(MODEL_DIR, "classical_dnn.pt")
        if os.path.exists(model_path):
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "classical_dnn_scaler.joblib"))
            self.n_features = joblib.load(os.path.join(MODEL_DIR, "classical_dnn_n_features.joblib"))
            self.model = _DNNNet(self.n_features).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            metrics_path = os.path.join(MODEL_DIR, "classical_dnn_metrics.joblib")
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
        self.n_features = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = _DNNNet(self.n_features).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
            logits = model(X_test_t)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        self.metrics = {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        }

        self.model = model
        self.scaler = scaler

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "classical_dnn.pt"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "classical_dnn_scaler.joblib"))
        joblib.dump(self.n_features, os.path.join(MODEL_DIR, "classical_dnn_n_features.joblib"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "classical_dnn_metrics.joblib"))
        self._loaded = True

    def predict(self, input_data: dict) -> dict:
        self.ensure_loaded()

        df = pd.read_csv(DATASET_PATH)
        drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        df.drop(columns=["diagnosis"], inplace=True, errors="ignore")
        feature_names = list(df.columns)
        medians = df.median()

        row = np.array([float(input_data.get(f, medians.get(f, 0))) for f in feature_names]).reshape(1, -1)
        scaled = self.scaler.transform(row)

        def _pred():
            t = torch.tensor(scaled, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logit = self.model(t).item()
                prob = float(torch.sigmoid(torch.tensor(logit)))
            label = "Malignant" if prob >= 0.5 else "Benign"
            confidence = prob if prob >= 0.5 else 1 - prob
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
