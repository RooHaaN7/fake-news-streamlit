import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import pipeline

# Load model from Hugging Face (replace with correct repo path)
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

# Streamlit UI
st.title("📰 Fake News Detector")
user_input = st.text_area("Enter News Article Text", height=200)

if st.button("Check"):
    if user_input.strip():
        result = pipe(user_input)[0]
        label = result['label']
        score = result['score']
        st.success(f"Prediction: **{label}** with confidence **{score:.2f}**")
    else:
        st.warning("Please enter some text.")
