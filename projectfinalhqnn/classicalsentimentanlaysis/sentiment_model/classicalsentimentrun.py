import os
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==================== LOAD MODEL ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_DIR

print(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")


# ==================== LOAD METRICS ====================
try:
    with open(os.path.join(MODEL_PATH, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    val_acc = metrics.get("validation_results", {}).get("eval_accuracy")
    test_acc = metrics.get("test_accuracy")
    print("\nModel Metrics:")
    if val_acc is not None:
        print(f"- Validation accuracy: {val_acc:.2%}")
    if test_acc is not None:
        print(f"- Test accuracy: {test_acc:.2%}")
except Exception:
    print("\nNo readable metrics.json found")


# ==================== PREDICTION FUNCTION ====================
def predict_sentiment(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)

    label = torch.argmax(probs).item()

    return {
        "sentiment": "Positive" if label == 1 else "Negative",
        "confidence": float(probs[0][label])
    }


# ==================== INTERACTIVE MODE ====================
print("\nType a review (or 'exit' to quit)\n")

while True:
    try:
        text = input("Enter review: ")
    except EOFError:
        print("\nExiting...")
        break

    if text.lower() == "exit":
        print("Exiting...")
        break

    result = predict_sentiment(text)

    print("\nPrediction:")
    print("Sentiment :", result["sentiment"])
    print("Confidence:", round(result["confidence"], 4))
    print("-" * 40)