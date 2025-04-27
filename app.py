import streamlit as st
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
import hashlib
import json
import os

# --- Page config ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")
# --- Theme Switcher ---
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme_choice = st.radio(
    "Choose Theme",
    ["Light", "Dark"],
    horizontal=True,
    index=0 if st.session_state.theme == "Light" else 1,
)

st.session_state.theme = theme_choice


# --- Password hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- User management ---
def load_users():
    if not os.path.exists("users.json"):
        with open("users.json", "w") as f:
            json.dump({}, f)
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def authenticate(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

# --- Session setup ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Load model ---
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# --- Light Mode CSS ---
light_mode_css = """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
            color: #212529;
        }
        .stMarkdown, .stText, .stTextArea textarea, .stButton > button, .stRadio label {
            color: #212529 !important;
        }
        [data-testid="stHeader"] {
            background-color: transparent;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #212529 !important;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #212529 !important;
            border: 1px solid #ced4da;
        }
        .stButton > button {
            background-color: #f1f1f1 !important;
            color: #212529 !important;
            border: 1px solid #ced4da !important;
        }
        .result-card {
            background-color: #ffffff;
            color: #212529;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            margin-top: 1rem;
        }
        .confidence-high {
            color: #198754;
        }
        .confidence-low {
            color: #dc3545;
        }
        .footer {
            text-align: center;
            color: #6c757d;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
        .navbar {
            background-color: #343a40;
            padding: 1rem 2rem;
            color: #ffffff;
            font-size: 1.25rem;
            font-weight: 600;
            text-align: center;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
    </style>
"""
st.markdown(light_mode_css, unsafe_allow_html=True)

# --- Navbar ---
st.markdown('<div class="navbar">📰 Real and Fake News Detection</div>', unsafe_allow_html=True)

# --- Login/Signup if not logged in ---
if not st.session_state.logged_in:
    st.title("🔐 Login / Sign Up")
    auth_mode = st.radio("Choose an option", ["Login", "Sign Up"], horizontal=True)

    if auth_mode == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials")
    else:
        new_username = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        if st.button("Sign Up"):
            if register_user(new_username, new_password):
                st.success("🎉 Account created! Please log in.")
            else:
                st.error("⚠️ Username already exists. Try another.")

# --- Main App if logged in ---
else:
    pipe = load_model()

    st.markdown(f"<h1 style='text-align: center;'>🧠 Fake News Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Instantly verify whether a news article is real or fake using DistilBERT</p>", unsafe_allow_html=True)

    st.markdown(f"👋 Welcome, **{st.session_state.username}**", unsafe_allow_html=True)

    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

    st.markdown("### 📝 Enter a News Article or Statement")
    user_input = st.text_area("Paste the news article below", height=200, label_visibility="collapsed")

    with st.expander("📌 Try an Example"):
        st.code("NASA has confirmed the moon is indeed made of cheese after astronauts discovered dairy-rich samples on their latest mission.")

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

            result_html = f"""
                <div class="result-card">
                    <h3>📢 Prediction Result</h3>
                    <p><strong>Label:</strong> {label_mapped}</p>
                    <p><strong>Confidence:</strong> <span class="{score_color}">{score:.2%}</span></p>
                </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

            if score < 0.6:
                st.warning(f"⚠️ Low confidence prediction. Confidence: {score:.2%}")
            elif label == "LABEL_1":
                st.success(f"✅ This article appears REAL with {score:.2%} confidence.")
            else:
                st.error(f"🚨 This article appears FAKE with {score:.2%} confidence.")

            with st.expander("🧬 Model Details"):
                st.markdown("- Model: `rohanN07/fake-news`")
                st.markdown("- Base: DistilBERT")
                st.markdown("- Fine-tuned for binary classification (REAL vs FAKE)")

            with st.expander("📘 Label Meaning"):
                st.markdown("**✅ REAL**: Likely factual and reliable.")
                st.markdown("**❌ FAKE**: Possibly misleading or incorrect.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    st.markdown("<div class='footer'>🔗 Built with HuggingFace Transformers & Streamlit · © 2025</div>", unsafe_allow_html=True)
