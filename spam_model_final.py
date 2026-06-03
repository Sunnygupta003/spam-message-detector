import pandas as pd
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join("dataset", "spam.csv")
MODEL_OUT_PATH = os.path.join("project", "model.pkl")
VEC_OUT_PATH   = os.path.join("project", "vectorizer.pkl")

print("Loading dataset from:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    # Try alternate path if not found (in case running from inside project/)
    DATA_PATH = os.path.join("..", "dataset", "spam.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check your folder structure.")

# Load dataset with correct encoding
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, errors='ignore')
df.columns = ["label", "sms"]

print(f"Dataset Loaded ✅ ({len(df)} rows)")

# Map labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Improved cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Basic whitespace cleanup
    text = " ".join(text.split())
    return text

df["sms"] = df["sms"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["sms"],
    df["label"],
    test_size=0.15, 
    random_state=42,
    stratify=df["label"]
)

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),     # single + word pairs
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# Model Training
model = MultinomialNB(alpha=0.2)
model.fit(X_train_vec, y_train)

print("Model trained ✅")

# Evaluation
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
print(f"\n📊 Model Performance: {acc*100:.2f}% Accuracy")

# Saving
print("\nSaving model files...")
os.makedirs("project", exist_ok=True)

with open(MODEL_OUT_PATH, "wb") as f:
    pickle.dump(model, f)
with open(VEC_OUT_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"✅ Saved to {MODEL_OUT_PATH}")
print(f"✅ Saved to {VEC_OUT_PATH}")

print("\nModel is ready for use in the web app!")
