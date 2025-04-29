# 📰 Fake News Detection App

This is a Streamlit-based web app that uses a fine-tuned DistilBERT model to detect whether a news article is **REAL** or **FAKE**. The model is hosted on Hugging Face under [`rohanN07/fake-news`](https://huggingface.co/rohanN07/fake-news).

## 🚀 Features

- 🔐 Login/Signup functionality
- 🌗 Light/Dark theme switcher
- 🧠 Text classification using `DistilBERT`
- 📊 Confidence score visualization
- 🔄 Streamlit caching for performance

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- `rohanN07/fake-news` model
- `AutoTokenizer` + `AutoModelForSequenceClassification`

## 🛠️ Setup

### For Hugging Face Spaces:

1. **Upload the following files:**
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `users.json` (an empty `{}` JSON file if you want to start fresh)

2. **Example `requirements.txt`:**

