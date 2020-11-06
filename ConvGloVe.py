#Convolutional Layers + Glove Embedding
#https://keras.io/examples/nlp/pretrained_word_embeddings/

import pickle
from numpy import array
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers
import tensorflow as tf 
import keras


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])


from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(tr_data_df['Tweet_Text']).batch(128)
vectorizer.adapt(text_ds)


#Convert to Numpy Vectors
train = vectorizer(np.array([[s] for s in tr_data_df['Tweet_Text']])).numpy()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
label = lb.fit_transform(np.array(tr_data_df["Party"]))

#USE Train_test_split for consistency instead of KFold. 
"""from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(train):
    x_train, x_val = train[train_index], train[test_index]
    y_train, y_val = label[train_index], label[test_index]"""

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2)

print(x_train)
print(y_train)

#Word Embedding
tokenizer = Tokenizer(num_words=5000)
vect=Tokenizer()
vect.fit_on_texts(tr_data_df['Tweet_Text'])
vocab_size = len(vect.word_index) + 1

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

#embeddings_index = {}
with open('./gloVe_vectors.obj', 'rb') as gloVe_vectors: 
    embeddings_index = pickle.load(gloVe_vectors)

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

int_sequences_input = layers.Input(shape=(None,), dtype="float64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(tr_data_df['Tweet_Text']), activation="softmax")(x)
model = tf.keras.Model(int_sequences_input, preds)
model.summary()

#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

model.save('./models/ConvGloVe.h5')