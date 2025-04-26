import streamlit as st
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
import hashlib
import json
import os

# ---------- Utility Functions ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

# ---------- Session State ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------- Model Loader ----------
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# ---------- Logged Out Interface ----------
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

# ---------- Logged In Interface ----------
else:
    pipe = load_model()
    
    st.markdown(f"### 👋 Welcome, **{st.session_state.username}**")
    st.markdown("Use the box below to classify a news article as real or fake.")

    if st.button("🔓 Logout", use_container_width=True):
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
            score_color = "🟢" if score >= 0.6 else "🔴"

            st.markdown(f"### 📢 Prediction Result")
            st.write(f"**Label:** {label_mapped}")
            st.write(f"**Confidence:** {score_color} {score:.2%}")

            if score < 0.6:
                st.warning(f"⚠️ Low confidence prediction.")
            elif label == "LABEL_1":
                st.success(f"✅ This article appears REAL.")
            else:
                st.error(f"🚨 This article appears FAKE.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
