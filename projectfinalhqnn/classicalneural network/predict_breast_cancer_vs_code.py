# ==============================
# Breast Cancer Prediction (Full 30 features input)
# ==============================

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import joblib
import os
import json

# Compatibility patch for Keras model configs saved with
# `quantization_config` in Dense layers.
_original_dense_from_config = Dense.from_config


def _dense_from_config_compat(config):
    if isinstance(config, dict):
        config = dict(config)
        config.pop("quantization_config", None)
    return _original_dense_from_config(config)


Dense.from_config = classmethod(lambda cls, config: _dense_from_config_compat(config))


def load_seed_metrics(metrics_file, main_seed=42):
    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file, "r", encoding="utf-8") as f:
        runs = json.load(f)

    if not isinstance(runs, list):
        return None

    for run in runs:
        if isinstance(run, dict) and run.get("seed") == main_seed:
            return run
    return None

# -----------------------------
# 1️⃣ Load trained model (.keras)
# -----------------------------
model_file = 'breast_cancer_model.keras'  # <- place in same folder
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file '{model_file}' not found.")

model = load_model(model_file)
print(f"Loaded Keras model: {model.name}")

# -----------------------------
# 2️⃣ Load saved scaler + PCA
# -----------------------------
scaler_pca_file = 'scaler_pca.pkl'  # <- place in same folder
if not os.path.exists(scaler_pca_file):
    raise FileNotFoundError(f"Scaler/PCA file '{scaler_pca_file}' not found.")

scaler, pca = joblib.load(scaler_pca_file)
print("Loaded scaler + PCA successfully.")

# -----------------------------
# 2.5️⃣ Show saved main metrics (seed 42)
# -----------------------------
metrics_file = "all_run_metrics.json"
main_metrics = load_seed_metrics(metrics_file, main_seed=42)
if main_metrics:
    print("\nMain reference metrics (seed 42):")
    print(f"- Accuracy: {main_metrics.get('accuracy', 0.0):.2%}")
    print(f"- F1 (macro): {main_metrics.get('f1_macro', 0.0):.4f}")
    print(f"- AUROC: {main_metrics.get('auroc', 0.0):.4f}")
    print(f"- MCC: {main_metrics.get('mcc', 0.0):.4f}")
    print(f"- Loss: {main_metrics.get('loss', 0.0):.4f}")
else:
    print(f"Note: '{metrics_file}' found no entry for seed 42 (or file missing).")

# -----------------------------
# 3️⃣ Define number of original features
# -----------------------------
n_features_original = 30  # Original WDBC features
n_features_pca      = 6   # PCA-reduced features for model input

# Running confusion-matrix counters for optional live metrics
tp = tn = fp = fn = 0


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _print_running_metrics(tp_count, tn_count, fp_count, fn_count):
    total = tp_count + tn_count + fp_count + fn_count
    accuracy = _safe_div(tp_count + tn_count, total)
    precision = _safe_div(tp_count, tp_count + fp_count)
    recall = _safe_div(tp_count, tp_count + fn_count)
    f1_score = _safe_div(2 * precision * recall, precision + recall)

    print(
        f"Running metrics from {total} labeled case(s): "
        f"Accuracy={accuracy:.2%}, F1={f1_score:.4f}"
    )

# -----------------------------
# 4️⃣ Interactive input
# -----------------------------
print(f"\nPlease enter {n_features_original} WDBC feature values separated by commas.")
print("Type 'quit' to exit.\n")
print("Example input (first 5 features only shown):")
print("17.99,10.38,122.8,1001,0.1184,... (total 30 values)\n")

while True:
    user_input = input("Enter patient data (or 'quit'): ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Exiting prediction engine.")
        break
    
    try:
        # Convert input string to float array
        input_list = [float(x.strip()) for x in user_input.split(',')]
        if len(input_list) != n_features_original:
            raise ValueError(f"Expected {n_features_original} values, got {len(input_list)}.")
        
        # Convert to 2D array for model
        input_arr = np.array(input_list).reshape(1, -1)
        
        # Apply the saved scaler
        input_scaled = scaler.transform(input_arr)
        
        # Apply PCA transformation
        input_pca = pca.transform(input_scaled)
        
        # Predict probability
        prob = model.predict(input_pca, verbose=0)[0][0]
        
        # Show prediction
        if prob > 0.5:
            print(f"──► PREDICTION : MALIGNANT  (confidence: {prob:.2%})")
        else:
            print(f"──► PREDICTION : BENIGN     (confidence: {1-prob:.2%})")
        print(f"Raw malignant probability: {prob:.4f}")
        if main_metrics:
            print(
                "Reference (seed 42) -> "
                f"Accuracy: {main_metrics.get('accuracy', 0.0):.2%}, "
                f"F1: {main_metrics.get('f1_macro', 0.0):.4f}"
            )

        # Optional true label entry for online Accuracy/F1 tracking.
        # 1 = malignant, 0 = benign. Press Enter to skip.
        true_label_input = input(
            "Optional true label? (1=malignant, 0=benign, Enter=skip): "
        ).strip()

        if true_label_input:
            if true_label_input not in {"0", "1"}:
                raise ValueError("True label must be 0, 1, or empty.")

            y_true = int(true_label_input)
            y_pred = 1 if prob > 0.5 else 0

            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            else:
                fn += 1

            _print_running_metrics(tp, tn, fp, fn)

        print()
        
    except ValueError as ve:
        print(f"[Input Error] {ve} — please recheck and try again.\n")
    except Exception as e:
        print(f"[Unexpected Error] {e}\n")