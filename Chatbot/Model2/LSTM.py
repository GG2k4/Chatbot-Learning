from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, Bidirectional, Dense, LSTM
from keras._tf_keras.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(oov_token = '<OOV>')

model = Sequential([
    Embedding(tokenizer.vocab_size, 64),
    Bidirectional(LSTM(64, return_sequences = True)),
    Bidirectional(LSTM(32)), 
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])