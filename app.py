import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

sentiment_mapping = {
    'Positive': 0,
    'Negative': 1,
    'Neutral': 2,
    'Irrelevant': 3
}

reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="wide")

st.title("Sentiment Analysis App")
st.write("""
This app predicts the sentiment of text using a pre-trained model.
Enter your text below, and the app will classify it as Positive, Negative, Neutral, or Irrelevant.
""")

@st.cache_resource  
def load_model_and_vectorizer():
    model = joblib.load("rf_model.joblib") 
    vectorizer = joblib.load("tfidf_vectorizer.joblib") 
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.subheader("Enter Text for Sentiment Prediction")
new_text = st.text_area("Input your text here:", "I love this product!")

if st.button("Predict Sentiment"):
    new_text_vec = vectorizer.transform([new_text])

    prediction_numeric = model.predict(new_text_vec)[0]
    prediction_label = reverse_sentiment_mapping[prediction_numeric]

    st.write(f"**Predicted Sentiment:** {prediction_label}")

    if hasattr(model, "predict_proba"):
        st.subheader("Prediction Probabilities")
        probabilities = model.predict_proba(new_text_vec)[0]
        prob_df = pd.DataFrame({
            "Sentiment": list(sentiment_mapping.keys()),
            "Probability": probabilities
        })
        st.bar_chart(prob_df.set_index("Sentiment"))

st.markdown("---")
st.write("Built with ❤️ by Saged Ahmed")