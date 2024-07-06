from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, Bidirectional, Dense, LSTM
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
import numpy as np

tokenizer = Tokenizer()
data = open("./irish_lyrics.txt").read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sentences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sentences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sentences])
input_sentences = np.array(pad_sequences(input_sentences, maxlen = max_sequence_len, padding = 'pre'))

xs, labels = input_sentences[:, :-1], input_sentences[:, -1]
ys = to_categorical(labels, num_classes = total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length = max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation = 'softmax'))
adam = Adam(learning_rate = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
history = model.fit(xs, ys, epochs = 100, verbose = 1)

seed_text = "I've got a bad feeling about this"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen = max_sequence_len - 1, padding = 'pre')
    predicted = np.argmax(model.predict(token_list), axis = -1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)