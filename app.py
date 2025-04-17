import streamlit as st
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")
# --- Theme Toggle ---
theme = st.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True)


# Load model
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

# --- Theme Toggle ---
theme = st.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True)


# --- Title ---
st.markdown("<h1 style='text-align: center;'>🧠 Fake News Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Instantly verify whether a news article is real or fake using DistilBERT</p>", unsafe_allow_html=True)

# --- Input Box ---
st.markdown("### 📝 Enter a News Article or Statement")
user_input = st.text_area("Paste the news article below", height=200, label_visibility="collapsed")

# --- Example Text ---
with st.expander("📌 Try an Example"):
    st.code("NASA has confirmed the moon is indeed made of cheese after astronauts discovered dairy-rich samples on their latest mission.")

# --- Analyze Button ---
if st.button("🚀 Analyze Text", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = pipe(user_input)[0]

        label_map = {
            "LABEL_0": "❌ FAKE",
            "LABEL_1": "✅ REAL"
        }

        label = result["label"]
        score = result["score"]
        label_mapped = label_map.get(label, label)
        score_color = "confidence-high" if score >= 0.6 else "confidence-low"

        # --- Result Card ---
        result_html = f"""
            <div class="result-card">
                <h3>📢 Prediction Result</h3>
                <p><strong>Label:</strong> {label_mapped}</p>
                <p><strong>Confidence:</strong> <span class="{score_color}">{score:.2%}</span></p>
            </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

        # --- Additional Feedback ---
        if score < 0.6:
            st.warning(f"⚠️ Low confidence prediction. Confidence: {score:.2%}")
        elif label == "LABEL_1":
            st.success(f"✅ This article appears REAL with {score:.2%} confidence.")
        else:
            st.error(f"🚨 This article appears FAKE with {score:.2%} confidence.")

        # --- Expanders ---
        with st.expander("🧬 Model Details"):
            st.markdown("- Model: `rohanN07/fake-news`")
            st.markdown("- Base: DistilBERT")
            st.markdown("- Fine-tuned for binary classification (REAL vs FAKE)")

        with st.expander("📘 Label Meaning"):
            st.markdown("**✅ REAL**: Likely factual and reliable.")
            st.markdown("**❌ FAKE**: Possibly misleading or incorrect.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# --- Footer ---
st.markdown("<div class='footer'>🔗 Built with HuggingFace Transformers & Streamlit · © 2025</div>", unsafe_allow_html=True)
