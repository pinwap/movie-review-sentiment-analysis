import os
import re
import string
import pickle

import numpy as np
import tensorflow as tf
import nltk

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# -----------------------
# Configuration
# -----------------------
output_dir       = "output"
model_filename   = "LSTM_text_classification.h5"
tokenizer_fname  = "tokenizer.pkl"
max_len          = 200

# -----------------------
# Download NLTK Data
# -----------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------
# Text Preprocessing
# -----------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# -----------------------
# Load Model & Tokenizer
# -----------------------
model_path     = os.path.join(output_dir, model_filename)
tokenizer_path = os.path.join(output_dir, tokenizer_fname)

# load the trained Keras model
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# load the fitted tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print(f"Loaded tokenizer from {tokenizer_path}")

# -----------------------
# Inference on New Samples
# -----------------------
def predict_sentiment(texts):
    # preprocess
    clean_texts = [preprocess_text(t) for t in texts]
    # tokenize & pad
    seqs = tokenizer.texts_to_sequences(clean_texts)
    pad  = pad_sequences(seqs, maxlen=max_len)
    # predict probabilities
    probs = model.predict(pad)
    # convert to binary labels
    labels = (probs > 0.5).astype(int)
    return probs.flatten(), labels.flatten()

if __name__ == "__main__":
    samples = [
        "I absolutely loved this movie—stellar acting and a great plot!",
        "Terrible. It was a waste of time, I almost fell asleep.",
        "An average film: some parts were good but overall forgettable."
    ]

    probs, preds = predict_sentiment(samples)
    for text, p, lbl in zip(samples, probs, preds):
        sentiment = "Positive" if lbl == 1 else "Negative"
        print(f"Review: {text}\n → P(positive) = {p:.3f}, Predicted: {sentiment}\n")
