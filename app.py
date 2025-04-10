import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")
    return model, tokenizer

model, tokenizer = load_model()

# App title
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below and the model will tell you if it's **REAL** or **FAKE**.")

# Text input
news_text = st.text_area("Paste the news article content below:", height=250)

# Predict button
if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some content to analyze.")
    else:
        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            confidence = torch.softmax(outputs.logits, dim=1).max().item()

        if prediction == 1:
            st.success(f"✅ This looks like **REAL** news. (Confidence: {confidence:.2%})")
        else:
            st.error(f"🚨 This looks like **FAKE** news. (Confidence: {confidence:.2%})")
<PASTE YOUR FULL app.py CODE HERE>
