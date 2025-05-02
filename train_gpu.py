import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
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

# Get every sequence and it's next character from the input data
sequences = []
next_chars = []
for i in range(0, len(text) - SEQ_LENGTH):
    sequences.append(text[i:i+SEQ_LENGTH])
    next_chars.append(text[i+SEQ_LENGTH])
# Output the number of sequences
print(f"Number of sequences: {len(sequences)}")

# Get the X and y training data, where X and y are series of one-hot encoded representations of the input sequence
X = np.zeros((len(sequences), SEQ_LENGTH, vocab_size), dtype=np.bool_)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)
for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Train the model with the GPU
with tf.device('/GPU:0'):
    # The model is a sequential model with Long Short-Term Memory (LSTM) Layers and Droput layers and a Softmax layer at the end
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, vocab_size)))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile the model and run it with categorical crossentropy loss and the adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    # Save the model to a keras file
    model.save(MODEL_FILE_PATH)