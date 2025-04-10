import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
    tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

pipe = load_model()

st.title("📰 Fake News Detector")
user_input = st.text_area("📝 Paste your news article text below:", height=200)

if st.button("🚀 Analyze"):
    if user_input.strip():
        result = pipe(user_input)[0]

        label_map = {
            "LABEL_0": "❌ FAKE",
            "LABEL_1": "✅ REAL"
        }

        label = result["label"]
        score = result["score"]
        label_mapped = label_map.get(label, label)

        st.markdown("### 🧠 Model Decision")
        st.json(result)

        st.markdown(f"**🔖 Raw Label:** `{label}`")
        st.markdown(f"**📊 Confidence Score:** `{score:.2%}`")

        if score < 0.6:
            st.warning(f"🧐 **Prediction:** {label_mapped} \n\n⚠️ Low confidence ({score:.2%}) — result may be unreliable.")
        else:
            if label == "LABEL_1":
                st.success(f"✅ This article looks **REAL** with {score:.2%} confidence.")
            else:
                st.error(f"🚨 This article appears **FAKE** with {score:.2%} confidence.")
    else:
        st.info("💡 Please enter some text above to analyze.")
