"""
Fake News Detector — Streamlit App
-----------------------------------
Loads the trained Bidirectional LSTM model + fitted Tokenizer and lets the
user paste a news article to get a Real / Fake prediction.

Expected files (place next to this script, or update MODEL_PATH / TOKENIZER_PATH):
    model.pkl
    Tokenzier.pkl
"""

import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "Model/model.pkl"
TOKENIZER_PATH = "Model/Tokenzier.pkl"
MAX_LEN = 300  # must match the value used during training

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# --------------------------------------------------------------------------
# Load model & tokenizer (cached so it only happens once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict(text: str, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probability = float(np.ravel(model.predict(padded, verbose=0))[0])
    label = "Real" if probability >= 0.5 else "Fake"
    confidence = probability if label == "Real" else 1 - probability
    return label, probability, confidence


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption(
    "A Bidirectional LSTM text classifier that estimates whether a news "
    "article is **Real** or **Fake** based on its text."
)

try:
    model, tokenizer = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False
    st.error(
        f"Could not find **{MODEL_PATH}** and/or **{TOKENIZER_PATH}**. "
        "Place both files in the same folder as this app, or update "
        "`MODEL_PATH` / `TOKENIZER_PATH` at the top of `app.py`."
    )

st.divider()

example_articles = {
    "— Select an example —": "",
    "Example: Wire-service style report": (
        "WASHINGTON (Reuters) - U.S. lawmakers said on Tuesday they had reached a "
        "bipartisan agreement on the annual defense spending bill, which now heads "
        "to a full vote in both chambers of Congress before the end of the month."
    ),
    "Example: Clickbait-style post": (
        "SHOCKING: You Won't Believe What This Celebrity Said About The Government! "
        "Click here to find out the SECRET they don't want you to know!!!"
    ),
}

choice = st.selectbox("Try an example, or paste your own text below:", list(example_articles.keys()))
default_text = example_articles[choice]

user_text = st.text_area(
    "Article text",
    value=default_text,
    height=220,
    placeholder="Paste the full text of a news article here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

if predict_clicked:
    if not artifacts_loaded:
        st.warning("Model files are missing — see the error above.")
    elif not user_text.strip():
        st.warning("Please enter or select some article text first.")
    else:
        with st.spinner("Analyzing article..."):
            label, probability, confidence = predict(user_text, model, tokenizer)

        st.divider()
        if label == "Real":
            st.success(f"### ✅ Likely **Real** News")
        else:
            st.error(f"### 🚩 Likely **Fake** News")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

        with st.expander("Details"):
            st.write(f"Raw model output (probability of *Real*): `{probability:.4f}`")
            st.write(
                "Prediction rule: probability ≥ 0.5 → **Real**, otherwise → **Fake**."
            )

st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown(
        """
- **Architecture:** Embedding (20,000 vocab, 128 dim) → Bidirectional LSTM (64 units) →
  Dropout → Dense(32, ReLU) → Dropout → Dense(1, Sigmoid)
- **Training data:** 5,829 labeled news articles
- **Test accuracy:** ~99.7% (see project report for important caveats about
  wire-service formatting leakage in the source dataset)
- **Disclaimer:** This is a research/educational prototype. It analyzes writing
  patterns and formatting — it does **not** fact-check claims. Always verify
  important news through trusted, independent sources.
        """
    )
