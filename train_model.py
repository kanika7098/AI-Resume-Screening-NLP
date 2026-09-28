import re, os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/Resume dataset.csv"
MODEL_PATH = "models/resume_classifier.joblib"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{10,}\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df = pd.read_csv(DATA_PATH)
df["clean_text"] = df["Text"].fillna("").apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["category"],
    test_size=0.20, random_state=42, stratify=df["category"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        sublinear_tf=True, max_features=50000, ngram_range=(1,2),
        min_df=2, max_df=0.95, stop_words="english"
    )),
    ("classifier", LinearSVC(C=1.5))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
with open("models/metrics.json", "w") as f:
    json.dump({"accuracy": float(acc), "classes": sorted(df["category"].unique().tolist())}, f, indent=2)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred))
print(f"Saved: {MODEL_PATH}")
