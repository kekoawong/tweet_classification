#https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
import pickle
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


with open('./data/testing_data.obj', 'rb') as testing_file:
    testing_data = pickle.load(testing_file)

with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])
tr_data_df
test_data_df = pd.DataFrame.from_records(testing_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])
test_data_df




# create our training data from the tweets
x = tr_data_df["Tweet_Text"]
# index all the sentiment labels
y = tr_data_df["Party"]


tokenizer = Tokenizer(num_words=5000)
vect=Tokenizer()
vect.fit_on_texts(x)
vocab_size = len(vect.word_index) + 1


X = tokenizer.texts_to_sequences(x.values)
X = pad_sequences(X, maxlen=250)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(y).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print("Train (X, Y):", X_train.shape,Y_train.shape)
print("test (X, Y):", X_test.shape,Y_test.shape)


#Keras
model = Sequential()
model.add(Embedding(50000, 100, input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])