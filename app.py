import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

st.set_page_config(page_title="Fake News Detector", layout="centered")

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")
    return model, tokenizer

model, tokenizer = load_model()

st.title("📰 Fake News Detection App")
st.markdown("Paste a news article below and we'll tell you if it's **FAKE** or **REAL**.")

news_text = st.text_area("Enter the news article:", height=250)

if st.button("🔍 Predict"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            conf = torch.softmax(outputs.logits, dim=1).max().item()

        if pred == 1:
            st.success(f"✅ REAL news (Confidence: {conf:.2%})")
        else:
            st.error(f"🚨 FAKE news (Confidence: {conf:.2%})")
