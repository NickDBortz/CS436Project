import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *

# Open the scripts file and read it all into a text variable
with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Create a sorted list of unique characters
chars = sorted(list(set(text)))
# Create maps from characters to indices and vice versa
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Get the total number of unique characters as the vocabulary size
vocab_size = len(chars)

# Load the model from the saved file
model = load_model(MODEL_FILE_PATH)

# Sample from the probability distribution with some temperature
def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Generate a certain length of text based on some starter seed and a temperature
def generate_text(seed, length, temperature):
    print(seed, end='')
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, vocab_size))
        for t, char in enumerate(seed[-SEQ_LENGTH:]):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        print(next_char, end='')
        seed += next_char

generate_text("INT. DARK ROOM - NIGHT\n", length=GENERATION_LENGTH, temperature=TEMPERATURE)