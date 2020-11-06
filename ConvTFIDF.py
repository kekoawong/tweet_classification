#Convolutional Layers + TFIDF
#https://www.kaggle.com/ismu94/tf-idf-deep-neural-net

#Sequential vs Functional API in Keras: https://stackoverflow.com/questions/58092176/keras-sequential-vs-functional-api-for-multi-task-learning-neural-network


import pandas as pd
import numpy as np
import pickle

#Load Training data/labels. 
with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])
tr_data_df
label = tr_data_df["Party"]

#Load Lemmatized Training data. 
with open('trained_lemmas.pk', 'rb') as data: 
    train = pickle.load(data)

#Binarize Labels
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
label = lb.fit_transform(np.array(label))

#Load Filtered and Fitted TFIDF Vectorizer. 
with open('vectorizer.pk', 'rb') as vector: 
    vectorizer = pickle.load(vector)
tf_len = len(vectorizer.vocabulary_)


#USE Train_test_split for consistency instead of KFold. 
"""from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(train):
    x_train, x_val = train[train_index], train[test_index]
    y_train, y_val = label[train_index], label[test_index]"""


#Split into Training and Testing Data. 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(tr_data_df['Tweet_Text'], label, test_size=0.3)


TF_X_train = vectorizer.transform(x_train).astype('float64').toarray()
TF_X_val = vectorizer.transform(x_val).astype('float64').toarray()


#Conv + TFIDF
#from keras.layers import Conv1D, MaxPooling1D, Dropout, Activation, Input, Embedding, Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras

import tensorflow as tf
#model = Sequential()

inputs = layers.Input(shape=(None,), dtype="float64")
x = layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(x_train), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
model.summary()



#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

model.fit(TF_X_train, y_train, batch_size=128, epochs=10, validation_data=(TF_X_val, y_val))

model.save('./models/ConvTFIDF.h5')