# 🧠 Twitter Sentiment Analysis with SVM

This project analyzes tweet sentiments using machine learning and provides a real-time prediction app built with Streamlit.

## 🚀 Project Overview
- Cleaned and preprocessed a large tweet dataset (neutral, negative, positive)
- Built and compared multiple ML models (Naive Bayes, Logistic Regression, SVM)
- Finalized a Linear SVM model with ~73.3% accuracy
- Deployed a real-time sentiment predictor with Streamlit

## 🔍 Features
- Custom tweet input box
- Live prediction output with emoji-based sentiment feedback
- Handles URLs, mentions, punctuation, and stopwords
- Trained on TF-IDF vectorized text

## 📁 Files Included
- `app.py` – Main Streamlit application
- `svm_sentiment_model.pkl` – Trained SVM classifier
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer for prediction
- `PRODIGY_DS_04.ipynb` – (Optional) Full training notebook

## 🛠️ Tech Stack
- Python 🐍
- Scikit-learn
- NLTK
- Streamlit
- Pandas, NumPy

## 💡 How to Run
1. Clone the repo:
    ```bash
    git clone https://github.com/gnanreddy11/PRODIGY_DS_04.git
    cd Twitter-Sentiment-Analysis
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```
