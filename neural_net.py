import json
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Declare Variables
vocab_size = 450
embedding_dim = 12
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 12
num_of_epochs = 130

class_names = ["Positive", "Negative", "Neutral"]

with open("dataset.json", "r") as f:
    datastore = json.load(f)

# Creating seperate lists for the sentences and the labels
sentences = []
labels = []

for item in datastore:
    sentences.append(item['comment'])
    labels.append(item['tag'])

# Creating the training and the testing data
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Creating the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Converting the texts to sequencies using the tokenizer
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Making the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

# Loading in the model
model = keras.models.load_model("comment_sentiment_model/")

# Compiling and fitting the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fitting the model
# history = model.fit(training_padded, training_labels, epochs=num_of_epochs,
#                     validation_data=(testing_padded, testing_labels), verbose=2)

# Getting the model summary
model.summary()

# Saving the model
model.save("comment_sentiment_model/")

# Reversing the word index
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_sentence(text):
    # Fancy Looking List Comprehension
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


sentence = ["Good video"]
# Getting the predictions to display
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length,
                       padding=padding_type, truncating=trunc_type)
# print(model.predict(padded))
prediction = model.predict(padded)

print(prediction)

# Getting the prediction and returning it mapped to the clsss names list defined above
response = class_names[np.argmax(prediction)]
print(np.argmax(prediction))
print(response)
