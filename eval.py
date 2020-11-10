
import pickle
import pandas as pd
import numpy as np
import joblib
import pickle5
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#***********************************
# DATA LOADING
#Import Testing Data For Prediction
with open('./data/testing_data.obj', 'rb') as testing_file:
    testing_data = pickle.load(testing_file)

test_data_df = pd.DataFrame.from_records(testing_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', "nth"])
test_data_df

with open("vectorizer.pk", 'rb') as pkl: 
    vect = pickle.load(pkl)
with open("test_lemmas.pk", 'rb') as pkl: 
    test_lemmas = pickle.load(pkl)

#Reload Data for SVM and NB
with open('trained_lemmas.pk', 'rb') as testing_file:
    read = testing_file.read()

trained_lemmas = pickle.loads(read)
trained_lemmas = pd.DataFrame(trained_lemmas)
print(trained_lemmas[0])

with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)
tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', 'nth'])
tr_data_df


# Create Test Data for NNs.
x = test_data_df["Tweet_Text"]
# index all the sentiment labels
y = test_data_df["Party"]
Y = pd.get_dummies(y).values

#Import Tokenizer
with open('./models/tokenizer.pickle', 'rb') as hand:
    loaded_tokenizer = pickle5.load(hand)
#***********************************

#***********************************
#KERAS DEEP LEARNING EVAL 


#Check ConvLSTM
print("Convolutional + LSTM")
model = load_model('./models/ConvBI-LSTM.h5')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

X = loaded_tokenizer.texts_to_sequences(x.values)
X = pad_sequences(X, maxlen=200)

predictions = model.predict(X)
predictions = (predictions> 0.5) 

print("ConvLSTM Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y.argmax(axis=1), predictions.argmax(axis=1), target_names=['Democrat','Republican'] )) #
print("Confusion Matrix:", confusion_matrix(Y.argmax(axis=1), predictions.argmax(axis=1))) #.argmax(axis=1)


#Check ConvGloVe
print("Convolutional + GloVe")
model = load_model('./models/ConvGloVe.h5')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


predictions = model.predict(X)
predictions = (predictions> 0.5) 

print("ConvGloVe Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions,target_names=['Democrat','Republican']))
print("Confusion Matrix:", confusion_matrix(Y.argmax(axis=1), predictions.argmax(axis=1)))


#Check ConvTFIDF
print("Convolutional + TFIDF")
model = load_model('./models/ConvTFIDF.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

predictions = model.predict(X)
predictions = (predictions> 0.5) 

print("ConvTFIDF Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions,target_names=['Democrat','Republican']))
print("Confusion Matrix:", confusion_matrix(Y.argmax(axis=1), predictions.argmax(axis=1)))

#***********************************

#***********************************
#SCIKIT LEARN MODELS EVAL

print("SVM")
Encoder = LabelEncoder()
Y =  Encoder.fit_transform(test_data_df["Party"])

model = joblib.load('models/SVM_TFIDF.joblib')
vect.fit(trained_lemmas[0])
x = vect.transform(test_lemmas)

predictions = model.predict(x)
predictions = (predictions> 0.5) 

print("SVM Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions, target_names=['Democrat','Republican'], zero_division="warn"))
print("Confusion Matrix:", confusion_matrix(Y, predictions))

print("NB")
model = joblib.load('models/NB_TFIDF.joblib')

predictions = model.predict(x)
predictions = (predictions> 0.5) 

print("NB Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions,target_names=['Democrat','Republican'], zero_division="warn"))
print("Confusion Matrix:", confusion_matrix(Y, predictions))

print("KNN")
model = joblib.load('models/KNN_TFIDF.joblib')

predictions = model.predict(x)
predictions = (predictions> 0.5) 

print("KNN Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions,target_names=['Democrat','Republican'], zero_division="warn"))
print("Confusion Matrix:", confusion_matrix(Y, predictions))


print("MLP")
model = joblib.load('models/MLP_TFIDF.joblib')

predictions = model.predict(x)
predictions = (predictions> 0.5) 

print("MLP Accuracy Score: ",accuracy_score(predictions, Y)*100)
print("Classification Report:", classification_report(Y, predictions,target_names=['Democrat','Republican'], zero_division="warn"))
print("Confusion Matrix:", confusion_matrix(Y, predictions))
#***********************************