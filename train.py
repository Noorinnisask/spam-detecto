import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    t = re.sub(r"[^a-z£$\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df = df.drop_duplicates().reset_index(drop=True)
df["clean"] = df["text"].apply(clean_text)
df["target"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2, stratify=df["target"], random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), target_names=["ham", "spam"]))
joblib.dump(model, "model.joblib")