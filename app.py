import re
import math
import joblib
import streamlit as st

st.set_page_config(page_title="Spam Detector", page_icon="📩")


def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    t = re.sub(r"[^a-z£$\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def spam_score(model, text):
    """Return a spam score between 0 and 1 for any model type."""
    x = [clean_text(text)]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[0][1]
    # LinearSVC: use decision_function and squash to 0-1
    return 1 / (1 + math.exp(-model.decision_function(x)[0]))


model = load_model()

st.title("📩 SMS Spam Detector")
st.write("Paste a message and the model will tell you if it looks like spam.")

threshold = st.sidebar.slider("Spam threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.caption(
    "Higher = fewer real messages blocked, but more spam gets through."
)

examples = {
    "(choose an example)": "",
    "Spam-like": "WINNER!! You have been selected for a free prize. Call now to claim your reward!",
    "Normal message": "Hey, are we still meeting for lunch tomorrow? Let me know.",
    "Tricky one": "Your parcel is waiting. Please confirm your address to get it delivered.",
}
choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Message", value=examples[choice], height=150)

if st.button("Check message"):
    if not text.strip():
        st.warning("Please type or paste a message first.")
    else:
        score = spam_score(model, text)
        if score >= threshold:
            st.error(f"🚨 Looks like SPAM (score: {score:.0%})")
        else:
            st.success(f"✅ Looks like a normal message (spam score: {score:.0%})")
        st.progress(float(score))
        st.caption(
            "The score is the model's estimate, not a guarantee. "
            "For Linear SVM it is an approximation, not a true probability."
        )

st.markdown("---")
st.caption("Built with scikit-learn (TF-IDF) and Streamlit.")