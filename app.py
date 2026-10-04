import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", page_icon="📰")


@st.cache_resource(show_spinner="Training model (first run only)...")
def train_model():
    df = pd.read_csv("news.csv")
    df = df.dropna(subset=["text", "label"])

    labels = df["label"].astype(str).str.strip().str.upper()
    label_map = {"FAKE": 0, "REAL": 1, "0": 0, "1": 1, "FALSE": 0, "TRUE": 1}
    df["label"] = labels.map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"]
    else:
        df["content"] = df["text"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy, len(df)


model, vectorizer, accuracy, n_articles = train_model()

# ---------- UI ----------
st.title("📰 Fake News Detector")
st.write("Paste a news article below and the model will predict whether it looks **real** or **fake**.")

text = st.text_area("News article text", height=250, placeholder="Paste the full article here...")

if st.button("Check article", type="primary"):
    if len(text.split()) < 20:
        st.warning("Please paste a longer article (at least 20 words) for a reliable result.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = proba[pred] * 100

        if pred == 1:
            st.success(f"✅ Looks REAL  (confidence: {confidence:.1f}%)")
        else:
            st.error(f"❌ Looks FAKE  (confidence: {confidence:.1f}%)")

        st.progress(float(proba[1]), text=f"Real: {proba[1]*100:.1f}%  |  Fake: {proba[0]*100:.1f}%")

with st.sidebar:
    st.header("Model info")
    st.write(f"**Articles used:** {n_articles:,}")
    st.write(f"**Test accuracy:** {accuracy*100:.2f}%")
    st.write("**Method:** TF-IDF + Logistic Regression")
    st.caption("This model learns writing patterns from one dataset. It does not verify facts, so treat its output as a demo, not a truth check.")