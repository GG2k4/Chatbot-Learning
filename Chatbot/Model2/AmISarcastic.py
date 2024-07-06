import json
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load data
with open("./sarcasm.json", 'r') as f:
    datastore = json.load(f)

# Extract data
sentences = [item['headline'] for item in datastore]
labels = [item['is_sarcastic'] for item in datastore]

# Split data
training_size = 20000
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = np.array(labels[:training_size])
testing_labels = np.array(labels[training_size:])

# Tokenization and padding
vocab_size = 10000  # Smaller vocab size for simplicity
oov_tok = '<OOV>'
padding_type = 'post'
trunc_type = 'post'
max_length = 100  # Set a max length for sequences
embedding_dim = 16  # Set embedding dimensions

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)

# Model definition
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# Train the model
history = model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=2)

# Testing new sentences
sentence = ["granny starting to fear spiders in the garden might be real", "the weather today is bright and sunny", "You realy see ghosts?"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
