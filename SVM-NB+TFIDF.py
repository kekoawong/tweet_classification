import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import nltk 
import pickle


#Additional Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', 'nth'])
tr_data_df

# create our training data from the tweets
#Load Lemmatized Training data. 
with open('trained_lemmas.pk', 'rb') as data: 
    x = pickle.load(data)
# index all the sentiment labels
y = tr_data_df["Party"]

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(x, y, test_size=0.3)



#Begin a for loops that will allow you to view 10 Naive Bayes and SVM accurracy scores and the mean of these accuacy scores 
#Before beginning the for loop we define variables for the mean_bayes_score and the mean_svm_score
mean_bayes_score = 0
mean_svm_score = 0 

#Step 5-- Prepare Train and Test Data Sets
#test_size specifies the size of the testing data set 
#add in the optional parameter random_state = 500 which is used to create reproducible outputs
#add in the optional parater stratify = Corpus['label'] which makes the split so that the proportion of
#values in the sample produced will be the same as the proportion of values provided 

#Step 6: Encoding -this step encodes the target labels with the value of 0 or 1
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#Step 7 -- Word Vectorization 
#max_features = 5000 --> builds a vocabulary that only considers the top max_fearures ordered by term frequency across the corpus
#Load Filtered and Fitted TFIDF Vectorizer. 
with open('vectorizer.pk', 'rb') as vector: 
    Tfidf_vect = pickle.load(vector)

#.fit learns vocabular and idf from training set 
Tfidf_vect.fit(x)

#.transform transforms documents to document-term matrix using the vocabulary and df learned by .fit
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#Step 8 --  Use the ML Algorithms to Predict the outcome

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# # Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score: ",accuracy_score(predictions_NB, Test_Y)*100)
# #OPTIONAL: Print a table with precision, recall, and F1 values for NB
print(classification_report(Test_Y, predictions_NB,target_names=['Democrat','Republican']))
# #OPTIONAL: prints the confusion matrix associated with the SVM predictions
print("NB Confusion Matrix: ", '\n', confusion_matrix(Test_Y, predictions_NB))


# #Add accuarcy score to mean_bayes_score 
# mean_bayes_score += accuracy_score(predictions_NB, Test_Y)*100

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma= 'auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validati on dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

print(predictions_SVM)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score: ",accuracy_score(predictions_SVM, Test_Y)*100)
# #OPTIONAL: prints a table with precision, recall, and F1 values
print(classification_report(Test_Y, predictions_SVM,target_names=['Democrat','Republican']))
# #OPTIONAL: prints the confusion matrix associated with the SVM predictions
print("SVM Confusion Matrix: ", '\n', confusion_matrix(Test_Y, predictions_SVM))

"""#Add accuarcy score to mean_svm_score 
mean_svm_score += accuracy_score(predictions_SVM, Test_Y)*100
print("\n")"""

#Additional Models
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Train_X_Tfidf,Train_Y)
predictions = knn.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("KNN Accuracy Score: ",accuracy_score(predictions, Test_Y)*100)
# #OPTIONAL: prints a table with precision, recall, and F1 values
print(classification_report(Test_Y, predictions,target_names=['Democrat','Republican']))
# #OPTIONAL: prints the confusion matrix associated with the SVM predictions
print("KNN Confusion Matrix: ", '\n', confusion_matrix(Test_Y, predictions))

mlp = MLPClassifier(random_state=1, max_iter=300)
mlp.fit(Train_X_Tfidf,Train_Y)
predictions = mlp.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("MLP Accuracy Score: ",accuracy_score(predictions, Test_Y)*100)
# #OPTIONAL: prints a table with precision, recall, and F1 values
print(classification_report(Test_Y, predictions,target_names=['Democrat','Republican']))
# #OPTIONAL: prints the confusion matrix associated with the SVM predictions
print("MLP Confusion Matrix: ", '\n', confusion_matrix(Test_Y, predictions))


#Print mean Bayes and SVM scores
print("Mean Bayes Score: ", mean_bayes_score/10)
print("Mean SVM Score: ", mean_svm_score/10)

#Model Persistance
from joblib import dump, load
dump(SVM, './models/SVM_TFIDF.joblib') 
dump(Naive, './models/NB_TFIDF.joblib') 
dump(knn, './models/KNN_TFIDF.joblib') 
dump(mlp, './models/MLP_TFIDF.joblib') 