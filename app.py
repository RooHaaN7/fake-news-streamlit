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
# Streamlit UI
st.markdown(
    "<h2 style='color:#4CAF50;'>🔎 Check if a News Article is Real or Fake</h2>",
    unsafe_allow_html=True,
)

user_input = st.text_area("📝 Paste your news article text below:", height=200)

st.markdown("---")

if st.button("🚀 Analyze"):
    if user_input.strip():
        result = pipe(user_input)[0]

        # Label map
        label_map = {
            "LABEL_0": "❌ FAKE",
            "LABEL_1": "✅ REAL"
        }

        label = result["label"]
        score = result["score"]
        label_mapped = label_map.get(label, label)

        # UI results
        st.markdown("### 🧠 Model Decision")
        st.json(result)

        st.markdown(f"**🔖 Raw Label:** `{label}`")
        st.markdown(f"**📊 Confidence Score:** `{score:.2%}`")

        if score < 0.6:
            st.warning(f"🧐 **Prediction:** {label_mapped} \n\n⚠️ Low confidence ({score:.2%})")
        else:
            if label == "LABEL_1":
                st.success(f"✅ This article looks **REAL** with {score:.2%} confidence.")
            else:
                st.error(f"🚨 This article appears **FAKE** with {score:.2%} confidence.")
    else:
        st.info("💡 Please enter some text above to analyze.")
