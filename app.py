import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

# Title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by DistilBERT · Quickly classify news articles as REAL or FAKE</p>", unsafe_allow_html=True)
st.divider()

# Input box
st.markdown("### 📝 Enter a news article or statement:")
user_input = st.text_area("", height=200, placeholder="Paste the news article content here...")

# Example Button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🎯 Use Example"):
        user_input = """NASA has confirmed the moon is indeed made of cheese after astronauts discovered dairy-rich samples on their latest mission."""

# Analyze button
if st.button("🚀 Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing... please wait"):
            result = pipe(user_input)[0]

        # Label mapping
        label_map = {
            "LABEL_0": "❌ FAKE",
            "LABEL_1": "✅ REAL"
        }
        label = result["label"]
        score = result["score"]
        label_mapped = label_map.get(label, label)

        # Display Results
        st.markdown("### 🧠 Model Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🔖 Label", value=label_mapped)
        with col2:
            st.metric(label="📊 Confidence", value=f"{score:.2%}")

        if score < 0.6:
            st.warning(f"⚠️ Low confidence prediction.\n\nPrediction: **{label_mapped}**\nConfidence: {score:.2%}")
        else:
            if label == "LABEL_1":
                st.success(f"✅ This article appears **REAL** with {score:.2%} confidence.")
            else:
                st.error(f"🚨 This article appears **FAKE** with {score:.2%} confidence.")

        # Expandable: Model Info
        with st.expander("🧬 Model Info"):
            st.markdown("- **Model:** `rohanN07/fake-news`")
            st.markdown("- **Architecture:** DistilBERT")
            st.markdown("- Fine-tuned for binary classification (REAL vs FAKE)")

        # Expandable: Label Explanation
        with st.expander("📘 Label Guide"):
            st.markdown("**✅ REAL**: The content is likely factual and trustworthy.")
            st.markdown("**❌ FAKE**: The content may be misleading, incorrect, or false.")
            st.info("Confidence over 60% is generally considered reliable.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.divider()
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using HuggingFace and Streamlit</p>", unsafe_allow_html=True)
