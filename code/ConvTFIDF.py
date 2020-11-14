#Convolutional Layers + TFIDF
#https://www.kaggle.com/ismu94/tf-idf-deep-neural-net
#Sequential vs Functional API in Keras: https://stackoverflow.com/questions/58092176/keras-sequential-vs-functional-api-for-multi-task-learning-neural-network
#Overfitting in TF: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit


import pandas as pd
import numpy as np
import pickle

#Load Training data/labels. 
with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', "nth"])
tr_data_df
label = tr_data_df["Party"]

#Load Lemmatized Training data. 
with open('trained_lemmas.pk', 'rb') as data: 
    train = pickle.load(data)


#Load Filtered and Fitted TFIDF Vectorizer. 
with open('vectorizer.pk', 'rb') as vector: 
    vectorizer = pickle.load(vector)
tf_len = len(vectorizer.vocabulary_)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
label = lb.fit_transform(np.array(tr_data_df["Party"]))

#USE Train_test_split for consistency instead of KFold. 
"""from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(train):
    x_train, x_val = train[train_index], train[test_index]
    y_train, y_val = label[train_index], label[test_index]"""

#Balance The Dataset
#https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
"""from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomnOverSampler
undersample = RandomOverSampler(sampling_strategy='minority')
train, label = oversample.fit_resample(train, label)"""

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2)

"""print(x_train)
print(y_train)"""

"""Encoder = preprocessing.LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_val = Encoder.fit_transform(y_val)"""


TF_X_train = vectorizer.transform(x_train).astype('float64').toarray()
TF_X_val = vectorizer.transform(x_val).astype('float64').toarray()


#Conv + TFIDF
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras

import tensorflow as tf

inputs = layers.Input(shape=(None,), dtype="float64")
x = layers.Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=64)(inputs)
x = layers.SpatialDropout1D(0.5)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.SpatialDropout1D(0.5)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.SpatialDropout1D(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)


history = model.fit(TF_X_train, y_train, batch_size=128, epochs=10, validation_data=(TF_X_val, y_val), callbacks=[es_callback], verbose=1)


model.save('./models/ConvTFIDF.h5')

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ConvTFIDF accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ConvTFIDFloss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()