import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Save and load vectorizer if needed

# Streamlit UI
st.title("Sentiment Prediction")
user_input = st.text_input("Enter a statement to predict its sentiment:")

if st.button('Predict Sentiment'):
    if user_input:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        st.write(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.write("Please enter a statement.")
