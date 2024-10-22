import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

choose_model = st.radio(
    "Choose your model:",
    ('SVM', 'Naive Bayes', 'Logistic Regression', 'LSTM'))


# Title and description
st.title(" Customer Product Reviews Sentiment Analysis App")


# Load your trained models
svc_model = joblib.load('svc_model.pkl')
naive_model = joblib.load('naive_model.pkl')
logistic_model = joblib.load('logistic_model.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
#nltk.download('stemming',quiet=True)

# Common stop words
stop_words = set(stopwords.words('english'))
stemming=PorterStemmer()





def clean_text(text):
    # 1. Convert to lower
    txt=text.lower()

    # 1. split to words
    tokens=word_tokenize(text)

    # 3. remove punctuation
    tokens=[word for word in tokens if word not in string.punctuation]

    # 4. Remove stopwords
    tokens=[word for word in tokens if word not in stop_words]

    # 5. Remove numbers
    tokens=[word for word in tokens if not word.isdigit()]

    # 6. Apply Stemming
    tokens=[stemming.stem(word) for word in tokens]

    # To return these single words back into one string
    return ' '.join(tokens)


# Text input for real-time feedback
user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type here...")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the text using the vectorizer
        user_input = clean_text(user_input)
        
        
        
        
        # Make the prediction and choose model
        if choose_model == 'SVM':
            prediction = svc_model.predict([user_input])
        elif choose_model == 'Naive Bayes':
            prediction = naive_model.predict([user_input])
        elif choose_model == 'Logistic Regression':
            prediction =logistic_model.predict([user_input])
        elif choose_model == 'LSTM':
            # Tokenize and pad the sequence for LSTM (assuming LSTM expects padded sequences)
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts([user_input])  # Fit only for this example
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=150)  # Assuming maxlen=100
            prediction = lstm_model.predict(padded_sequence)
            prediction = (prediction > 0.5).astype("int32")  # Convert probabilities to class labels

            
        # Debugging: Print the prediction to see the output format
        st.write(f"Raw prediction output: {prediction}")

        # Convert prediction to sentiment labels
        if choose_model != 'LSTM':
            # For non-LSTM models, prediction is typically a single value array (e.g., [1] or [0])
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
        else:
            # For LSTM, we handle binary class prediction
            sentiment = "Positive" if prediction[0][0] == 1 else "Negative"

        # Display sentiment with visualization
        if sentiment == "Positive":
            st.success(f"Prediction: {sentiment} 🩷")
        else:
            st.error(f"Prediction: {sentiment} 🥲")