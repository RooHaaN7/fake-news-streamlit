import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            border: 1px solid #ced4da;
            border-radius: 0.5rem;
        }
        .result-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            margin-top: 1rem;
        }
        .confidence-high {
            color: green;
        }
        .confidence-low {
            color: red;
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🧠 Fake News Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Instantly verify whether a news article is real or fake using DistilBERT</p>", unsafe_allow_html=True)

# --- Input Container ---
with st.container():
    st.markdown("### 📝 Input Article")
    user_input = st.text_area("Paste your news article below", height=200, label_visibility="collapsed")

    with st.expander("🔍 Need a test example?"):
        st.code("NASA has confirmed the moon is indeed made of cheese after astronauts discovered dairy-rich samples on their latest mission.")

# --- Analyze Button ---
if st.button("🚀 Analyze Text", use_container_width=True):
    if user_input.strip():
        with st.spinner("Processing with AI..."):
            result = pipe(user_input)[0]

        label_map = {
            "LABEL_0": "❌ FAKE",
            "LABEL_1": "✅ REAL"
        }

        label = result["label"]
        score = result["score"]
        label_mapped = label_map.get(label, label)
        score_color = "confidence-high" if score >= 0.6 else "confidence-low"

       # --- Result Display ---
st.markdown(f"""
<div class="result-card">
    <h3>📢 Prediction Result</h3>
    <p><strong>Label:</strong> {label_mapped}</p>
    <p><strong>Confidence:</strong> <span class="{score_color}">{score:.2%}</span></p>
</div>
""", unsafe_allow_html=True)

if score < 0.6:
    st.markdown("<p style='color: red;'>⚠️ Low confidence. The prediction might not be reliable.</p>", unsafe_allow_html=True)
elif label == "LABEL_1":
    st.success(f"✅ This article appears REAL with {score:.2%} confidence.")
else:
    st.error(f"🚨 This article appears FAKE with {score:.2%} confidence.")

