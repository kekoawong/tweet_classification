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

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', "nth"])

#https://swatimeena989.medium.com/bert-text-classification-using-keras-903671e0207d
from transformers import *
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig

bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)

input_ids = []
attention_masks = []
for sent in tr_data_df['Tweet_Text']:
    sequence = bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =280 ,pad_to_max_length = True,
                                            truncation=True, return_attention_mask = True)
    input_ids.append(sequence['input_ids'])
    attention_masks.append(sequence['attention_mask'])


tr_data_df['gt'] = tr_data_df['Party'].map({'Democrat':0,'Republican':1})
labels = tr_data_df['gt']

input_ids=np.array(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)

print(len(input_ids),len(attention_masks),len(labels))

#Pickle for later
"""print('Preparing the pickle file.....')

pickle_inp_path='./data/distilbert_inp.pkl'
pickle_mask_path='./data/distilbert_mask.pkl'
pickle_label_path='./data/distilbert_label.pkl'

pickle.dump((input_ids),open(pickle_inp_path,'wb'))
pickle.dump((attention_masks),open(pickle_mask_path,'wb'))
pickle.dump((labels),open(pickle_label_path,'wb'))


print('Pickle files saved as ',pickle_inp_path,pickle_mask_path,pickle_label_path)
"""

#Split into training and testing
from sklearn.model_selection import train_test_split
train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)
print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))


#Prepare loss and Optimizer
log_dir='tensorboard_data/tb_distilbert'
model_save_path='./models/distilbert_model.h5'

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)] #, keras.callbacks.TensorBoard(log_dir=log_dir),

print('\nBert Model',bert_model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

#TRAIN
history=bert_model.fit([train_inp,train_mask],train_label,batch_size=32,epochs=4,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)