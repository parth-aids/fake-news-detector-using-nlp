import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("news.csv")
print("Columns:", df.columns.tolist())
print("Label values:", df["label"].unique()[:10])

df = df.dropna(subset=["text", "label"])

# Convert labels to 0 (fake) / 1 (real), whatever format they're in
labels = df["label"].astype(str).str.strip().str.upper()
label_map = {"FAKE": 0, "REAL": 1, "0": 0, "1": 1, "FALSE": 0, "TRUE": 1}
df["label"] = labels.map(label_map)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Use the title too, if the file has one
if "title" in df.columns:
    df["content"] = df["title"].fillna("") + " " + df["text"]
else:
    df["content"] = df["text"]

print("\nTotal articles:", len(df))
print(df["label"].value_counts().rename({0: "Fake", 1: "Real"}))

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# 3. TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluate
preds = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=["Fake", "Real"]))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# 6. Most influential words
words = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
print("\nTop 'fake' words:", [words[i] for i in coefs.argsort()[:10]])
print("Top 'real' words:", [words[i] for i in coefs.argsort()[-10:]])

# 7. Try your own text
while True:
    text = input("\nPaste a news headline/article (or 'q' to quit): ")
    if text.lower() == "q":
        break
    pred = model.predict(vectorizer.transform([text]))[0]
    print("Prediction:", "Real ✅" if pred == 1 else "Fake ❌")