import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load pre-trained model
interpreter = tf.lite.Interpreter(model_path="sentiment_analysis.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the TfidfVectorizer used during model training (ensure the vectorizer matches the one used during training)
vectorizer = TfidfVectorizer()

# Get stopwords for preprocessing
stop_words = set(stopwords.words('english'))

# Tokenization and Stopword Removal Function
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric tokens
    return ' '.join(filtered_tokens)

# Streamlit interface
st.title('Sentiment Analysis using TensorFlow Lite')

user_input = st.text_area('Enter the statement to predict sentiment:')

if user_input:
    # Preprocess input (tokenization and vectorization)
    cleaned_input = tokenize_and_remove_stopwords(user_input)
    input_vector = vectorizer.transform([cleaned_input]).toarray()  # Convert to array for prediction

    # Run inference with TensorFlow Lite model
    input_data = np.array(input_vector, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the result (probabilities for each class)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Map the prediction to class labels (replace with your own labels)
    classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7']
    predicted_class = classes[prediction]

    st.write(f"The predicted sentiment is: {predicted_class}")
