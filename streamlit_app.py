import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load Dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv(r'https://docs.google.com/spreadsheets/d/e/2PACX-1vRKp00e2XC6CO64dGkVo8nOyD3FQgVjOho8W80U2L5XIsFyUFFl9_F8o8cburjfA20d2uw46q43Ei9l/pub?gid=1711183428&single=true&output=csv')

# Assuming dataset has two columns: 'text' and 'label'
X = data['statement']
y = data['status']
# Get stopwords
stop_words = set(stopwords.words('english'))

# Tokenization and Stopword Removal Function
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric tokens
    return ' '.join(filtered_tokens)
data['cleaned_text'] = X.fillna('').apply(tokenize_and_remove_stopwords)
# Vectorize Cleaned Text
vectorizer = TfidfVectorizer()  # No need to set stop_words here; already removed

X_vectorized = vectorizer.fit_transform(data['cleaned_text'])
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)
# Model Testing
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Function to Predict Sentiment
def predict_sentiment(user_input):
    processed_input = vectorizer.transform([user_input])
    prediction = model.predict(processed_input)
    return prediction[0]

# Streamlit UI
st.title("Sentiment Prediction")
user_input = st.text_input("Enter a statement to predict its sentiment:")

# Add a button to trigger the prediction
if st.button('Predict Sentiment'):
    if user_input:  # Only predict if there is input
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a statement to predict the sentiment.")

# Optionally, trigger prediction when the Enter key is pressed (streamlit default behavior)
if user_input:
    sentiment = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {sentiment}")
