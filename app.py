from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("rohanN07/fake-news")
tokenizer = DistilBertTokenizerFast.from_pretrained("rohanN07/fake-news")
