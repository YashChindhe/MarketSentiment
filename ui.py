import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Download stopwords
nltk.download('stopwords')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'\$[A-Za-z]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

file=pd.read_csv('stock_data.csv')

# Sample training data
texts = file.Text
labels = file.Sentiment  # 1 = positive, 0 = negative

# Clean text
clean_texts = [clean_text(t) for t in texts]

# Tokenize and pad
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(clean_texts)
sequences = tokenizer.texts_to_sequences(clean_texts)
padded = pad_sequences(sequences, maxlen=32, padding='post', truncating='post')

# Build model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64, input_length=32))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

# Train model (light training on dummy data for demo)
model.fit(padded, np.array(labels), epochs=5, verbose=0)

# Streamlit UI
st.title("📈 Financial Sentiment Analysis")
st.write("Enter a financial statement or market-related sentence to analyze its sentiment.")

user_input = st.text_area("📝 Type your financial news or statement here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=32, padding='post', truncating='post')
        pred = model.predict(pad)[0][0]
        sentiment = "🟢 Positive" if pred >= 0.5 else "🔴 Negative"
        st.subheader("Sentiment Result:")
        st.success(sentiment)
