# Market Sentiment Analysis (Streamlit)

This project performs sentiment analysis on financial or market-related text using a BiLSTM model built with TensorFlow Keras and a Streamlit user interface.

The model is trained from a CSV file **only once**.  
If a trained model already exists, it is loaded and reused to avoid retraining on every run.

---

## Project Files

```
MarketSentiment/
├── ui.py                # Streamlit application (main entry point)
├── stock_data.csv       # Training data (Text, Sentiment)
├── sentiment_model.h5   # Saved model (created automatically)
├── tokenizer.pkl        # Saved tokenizer (created automatically)
├── requirements.txt
└── README.md
```

---

## How It Works

1. The app checks if `sentiment_model.h5` and `tokenizer.pkl` exist.
2. If they exist:
   - The model and tokenizer are loaded.
3. If they do not exist:
   - Data is read from `stock_data.csv`.
   - Text is cleaned and tokenized.
   - A BiLSTM model is trained.
   - The model and tokenizer are saved for future use.
4. User input is analyzed using the loaded model.

---

## Input Data Format

The file `stock_data.csv` must contain the following columns:

| Column    | Description                     |
|----------|---------------------------------|
| Text     | Financial text or news sentence |
| Sentiment| 1 = Positive, 0 = Negative      |

---

## Run the Application

1. Install dependencies :
   ```pip install -r requirements.txt```
2. Run dashboard :
   ```streamlit run ui.py```

- First run: model is trained and saved.
- Next runs: saved model is loaded and reused.
- The app opens automatically in the browser.

---

## Results

<img width="2560" height="1440" alt="Screenshot (17)" src="https://github.com/user-attachments/assets/185f21ac-395f-4165-89fe-413c67a89306" />

---

## Notes

- This project performs sentiment classification, not stock price prediction.
- The tokenizer is saved because the model depends on the exact word-to-index mapping.
- Training inside the UI is done only once for simplicity and demo purposes.
