


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import nltk 
import pickle

import tqdm
from tqdm.contrib.concurrent import process_map

#Map Lemmatizer to tqdm process_map. 
def lemmatize_and_remove_stopwords(entry):
    # WordNetLemmatizer requires part of speech tags to understand if the word is noun or verb or adjective etc. 
    #By default it is set to Noun using the defaultdict 
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    # pos_tag returns a list of tuples
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    return str(Final_words)

def parrallel_tokenize(entry):
    return word_tokenize(str(entry))


with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)

tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])
tr_data_df

# create our training data from the tweets
tr_data_df["Tweet_Text"] = process_map(parrallel_tokenize, [entry for entry in tr_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1) 
train = process_map(lemmatize_and_remove_stopwords, [entry for entry in tr_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1)
# index all the sentiment labels
label = tr_data_df["Party"]

#max_features = 10000 --> builds a vocabulary that only considers the top max_fearures ordered by term frequency across the corpus
#vectorizer = TfidfVectorizer(max_features=10000)
vectorizer = TfidfVectorizer(max_features=1000) #use_idf=True)
vectorizer = vectorizer.fit(train)

#Pickle cleaned text TFIDF Vectorizer. 
with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

#Pickle Unfiltered text TFIDF Vectorizer. 
#Not tested yet.

#Pickle Lemmatized vectors.
with open('trained_lemmas.pk', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)