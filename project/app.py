"""
app.py  –  Fake Message Detector  (Flask backend)
Run:  python app.py
"""

import os
import pickle
import string
import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH      = os.path.join(BASE_DIR, "vectorizer.pkl")
EXCEL_PATH    = os.path.join(BASE_DIR, "data.xlsx")

# ─────────────────────────────────────────────────────────────
# Load pre-trained model & vectorizer
# ─────────────────────────────────────────────────────────────
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            m = pickle.load(f)
        with open(VEC_PATH, "rb") as f:
            v = pickle.load(f)
        return m, v
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

if model and vectorizer:
    print("✅ Model and vectorizer loaded successfully.")
else:
    print("⚠️ Warning: Model and/or vectorizer missing. Backend will return error on /predict.")

# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase + remove punctuation (must match training pipeline)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join(text.split())
    return text


def save_to_excel(message: str, prediction: str, confidence: float):
    """Append one row to data.xlsx, creating the file if needed."""
    row = {
        "Message":    message,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        if os.path.exists(EXCEL_PATH):
            df_existing = pd.read_excel(EXCEL_PATH, engine="openpyxl")
            df_new      = pd.DataFrame([row])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame([row])

        df_combined.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
    except Exception as e:
        print(f"Failed to save to Excel: {e}")


def get_history(n: int = 5) -> list[dict]:
    """Return the last *n* predictions from data.xlsx."""
    if not os.path.exists(EXCEL_PATH):
        return []
    try:
        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        tail = df.tail(n).iloc[::-1]  # most-recent first
        return tail.to_dict(orient="records")
    except Exception as e:
        print(f"Failed to read history: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer
    if model is None or vectorizer is None:
        # Retry loading once just in case they were just created
        model, vectorizer = load_models()
        if model is None:
            return jsonify({"error": "Model files not available on server."}), 500

    try:
        data    = request.get_json(force=True)
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "No message provided."}), 400

        cleaned  = clean_text(message)
        vec      = vectorizer.transform([cleaned])
        pred_int = model.predict(vec)[0]
        proba    = model.predict_proba(vec)[0]

        label      = "SPAM" if pred_int == 1 else "HAM"
        confidence = float(proba[pred_int]) * 100

        save_to_excel(message, label, confidence)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2),
            "is_spam":    bool(pred_int == 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history():
    records = get_history(10)
    return jsonify(records)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
