import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Constants
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 32
VOCAB_SIZE = 1000

# NLTK
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'\$[A-Za-z]+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join(word for word in text.split() if word not in STOP_WORDS)
    return text

# Load or Train Model
@st.cache_resource
def load_or_train_model():
    # IF model already exists → load
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer

    # Else, train once and save
    file = pd.read_csv("stock_data.csv")

    texts = file.Text
    labels = file.Sentiment

    clean_texts = [clean_text(t) for t in texts]

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(clean_texts)

    sequences = tokenizer.texts_to_sequences(clean_texts)
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(0.001),
        metrics=["accuracy"]
    )

    model.fit(padded, np.array(labels), epochs=5, verbose=0)

    # Save model
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer

# Load artifacts
model, tokenizer = load_or_train_model()

# Streamlit UI
st.title("Financial Sentiment Analysis")
st.write("Enter a financial statement or market-related sentence to analyze its sentiment.")

user_input = st.text_area("📝 Type your financial news or statement here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        pred = model.predict(padded, verbose=0)[0][0]
        sentiment = "Positive" if pred >= 0.5 else "Negative"

        st.subheader("Sentiment Result:")
        st.success(sentiment)