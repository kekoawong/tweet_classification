#Conv + LSTM
#https://www.tensorflow.org/tutorials/text/text_classification_rnn

import tensorflow as tf 
import keras
import pickle
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import re


with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])

#Vectorize Corpus
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

#Split into Training and Testing Data. 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2)

tokenizer = Tokenizer(num_words=5000)
vect=Tokenizer()
vect.fit_on_texts(tr_data_df['Tweet_Text'])
vocab_size = len(vect.word_index) + 1
with open('./models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Conv + LSTM
inputs = layers.Input(shape=(None,), dtype="float64")
x = layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64)(inputs)
x = layers.SpatialDropout1D(0.2)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.Bidirectional(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))(x)
outputs = layers.Dense(len(tr_data_df['Tweet_Text']), activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
model.save('./models/ConvBI-LSTM.h5')
