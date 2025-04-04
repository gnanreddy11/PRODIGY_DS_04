import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("svm_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing functions
stop_words = set(stopwords.words("english"))


def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])


def predict_sentiment(tweet):
    cleaned = remove_stopwords(clean_tweet(tweet))
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment_map = {0: "Negative 😡", 1: "Neutral 😐", 2: "Positive 😃"}
    return sentiment_map[prediction]


# Streamlit UI
st.title("Twitter Sentiment Analysis 🚀")
st.markdown("Enter a tweet below and find out its sentiment!")

user_input = st.text_area("Type your tweet here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {result}")
