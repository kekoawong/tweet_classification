#Conv + LSTM
#https://www.tensorflow.org/tutorials/text/text_classification_rnn
#Overfitting in TF: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit


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

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', 'nth'])

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
"""from sklearn.model_selection import KFoldr
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(train):
    x_train, x_val = train[train_index], train[test_index]
    y_train, y_val = label[train_index], label[test_index]"""

#Balance The Dataset
#https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
"""from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
train, label = undersample.fit_resample(train, label)"""

x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2)

"""y_train = tf.one_hot(y_train, depth=2)
y_val = tf.one_hot(y_val, depth=2)"""

print(x_train)
print(y_train)

"""Encoder = preprocessing.LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_val = Encoder.fit_transform(y_val)"""


tokenizer = Tokenizer(num_words=5000)
vect=Tokenizer()
vect.fit_on_texts(tr_data_df['Tweet_Text'])
vocab_size = len(vect.word_index) + 1
with open('./models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Conv + LSTM
#Use functional API approach. 
#Below is the sequential version of the same NN. 
#from tensorflow.keras.models import Sequential
"""model = Sequential()
model.add(layers.Input(shape=(None,), dtype="float64"))
model.add(layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64, input_length=200))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(64, 5, activation="relu"))
model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))"""

inputs = layers.Input(shape=(None,), dtype="float64")
x = layers.Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=64)(inputs)
x = layers.SpatialDropout1D(0.5)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.SpatialDropout1D(.5)(x) #Handle Potential Overfitting
x = layers.MaxPooling1D(5)(x)
x = layers.SpatialDropout1D(0.5)(x) #Handle Potential Overfitting
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.Bidirectional(layers.LSTM(100, dropout=0.5, recurrent_dropout=0.6))(x)
outputs = layers.Dense(2, activation='softmax')(x)
#Softmax: https://www.machinecurve.com/index.php/2020/01/08/how-does-the-softmax-activation-function-work/

model = tf.keras.Model(inputs, outputs)
model.summary()

#Overfitting: https://towardsdatascience.com/dont-overfit-how-to-prevent-overfitting-in-your-deep-learning-models-63274e552323


"""
Note the 3 changes:

loss function changed to categorical cross-entropy
No. of units in final Dense layer is 3
One-hot encoding of labels is required and can be done using tf.one_hot

tf.one_hot(train_labels, 3)
"""

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
""" 
Using the sigmoid activation function in the last layer is wrong because 
the sigmoid function maps logit values to a range between 0 and 1 (However, 
your class labels are 0, 1 and -1). This clearly shows that the network will 
never be able to predict a negative value because of the sigmoid function 
(which can only map values between 0 and 1) and hence, will never learn to predict 
the negative class. The right approach would be to view this as a multi-class classification 
problem and use the categorical cross-entropy loss accompanied by the softmax activation in your 
last Dense layer with 3 units (one for each class). Note that one-hot encoded labels have to be used 
for the categorical cross-entropy loss and integer labels can be used along with the sparse categorical 
cross-entropy loss.
"""

#import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
"""
#Overfitting: 
#https://towardsdatascience.com/dont-overfit-how-to-prevent-overfitting-in-your-deep-learning-models-63274e552323

3: Early Stopping
Another way to prevent overfitting is to stop your training process 
early: Instead of training for a fixed number of epochs, you stop as 
soon as the validation loss rises — because, after that, your model will 
generally only get worse with more training. You can implement early 
stopping easily with a callback in keras:
"""
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val), callbacks=[es_callback], verbose=1)
model.save('./models/ConvBI-LSTM.h5')

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LSTM accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


loss: 0.4352 - accuracy: 0.7904 - val_loss: 0.5153 - val_accuracy: 0.7438
loss: 0.2916 - accuracy: 0.8709 - val_loss: 0.5997 - val_accuracy: 0.7523
loss: 0.1736 - accuracy: 0.9278 - val_loss: 0.6931 - val_accuracy: 0.7473
