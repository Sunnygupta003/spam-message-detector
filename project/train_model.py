"""
train_model.py
Run this ONCE to generate model.pkl and vectorizer.pkl
Place spam.csv in the same directory before running.
"""

import os
import pickle
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "spam.csv")

print("Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# Keep only the first two columns
df = df.iloc[:, :2]
df.columns = ["label", "sms"]

print(f"Dataset loaded ✅  →  Rows: {len(df)}")

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ─────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["sms"] = df["sms"].apply(clean_text)

# ─────────────────────────────────────────────
# 3. Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["sms"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# ─────────────────────────────────────────────
# 4. Vectorise
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────
model = MultinomialNB(alpha=0.3)
model.fit(X_train_vec, y_train)

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
preds = model.predict(X_test_vec)

print("\n📊 Model Performance")
print("─" * 30)
print(f"Accuracy : {accuracy_score(y_test, preds)*100:.2f}%")
print(f"Precision: {precision_score(y_test, preds)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, preds)*100:.2f}%")
print(f"F1 Score : {f1_score(y_test, preds)*100:.2f}%")

# ─────────────────────────────────────────────
# 7. Save artefacts
# ─────────────────────────────────────────────
out_dir = os.path.dirname(__file__)

with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(out_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ model.pkl and vectorizer.pkl saved successfully!")
