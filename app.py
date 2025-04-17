import streamlit as st
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# --- Theme Toggle ---
theme = st.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"], horizontal=True, key="theme_toggle")

# Load model
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

# --- Light Mode CSS ---
light_mode_css = """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f8f9fa;
            color: #000000;
        }
        .stText, .stMarkdown, .stTextArea textarea, .stButton > button, .stRadio label {
            color: #000000 !important;
        }
        [data-testid="stHeader"] {
            background-color: transparent;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da;
        }
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
        }
        .result-card {
            background-color: #ffffff;
            color: #000000;
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
"""

# --- Dark Mode CSS ---
dark_mode_css = """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stText, .stMarkdown, .stTextArea textarea, .stButton > button, .stRadio label {
            color: #ffffff !important;
        }
        [data-testid="stHeader"] {
            background-color: transparent;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        .stTextArea textarea {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #444444;
        }
        .stButton > button {
            background-color: #222222 !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
        }
        .result-card {
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(255, 255, 255, 0.05);
            margin-top: 1rem;
        }
        .confidence-high {
            color: #00ff00;
        }
        .confidence-low {
            color: #ff4b4b;
        }
        .footer {
            text-align: center;
            color: #aaaaaa;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
"""

# Apply the selected theme
if theme == "🌞 Light Mode":
    st.markdown(light_mode_css, unsafe_allow_html=True)
else:
    st.markdown(dark_mode_css, unsafe_allow_html=True)

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
