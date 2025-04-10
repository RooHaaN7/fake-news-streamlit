import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import pipeline

# Load model from Hugging Face
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

       # Map labels
label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}
label = result["label"]
score = result["score"]
label_mapped = label_map.get(label, label)

st.write(f"🧠 Raw Label: {label}")
st.write(f"📊 Confidence: {score:.2%}")

# Optional: Add a threshold for certainty
if score < 0.6:
    st.warning(f"🧐 Prediction: **{label_mapped}**, but confidence is low ({score:.2%})")
else:
    st.success(f"✅ Prediction: **{label_mapped}** with confidence {score:.2%}")
