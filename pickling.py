


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

###Import Training Data
with open('./data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)
tr_data_df = pd.DataFrame.from_records(training_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', "nth"])
tr_data_df

with open('./data/training_data_ht_ment.obj', 'rb') as unfiltered: 
    ht_unfiltered = pickle.load(unfiltered)
unfiltered_data_df = pd.DataFrame.from_records(ht_unfiltered['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])

####Import Testing Data
with open('./data/testing_data.obj', 'rb') as training_file:
    test_data = pickle.load(training_file)
test_data_df = pd.DataFrame.from_records(test_data['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle', "nth"])
test_data_df

with open('./data/testing_data_ht_ment.obj', 'rb') as unfiltered: 
    ht_test = pickle.load(unfiltered)
unfiltered_test_df = pd.DataFrame.from_records(ht_test['tweets'], columns=['Party', 'Tweet_Text', 'Hashtags', 'Mentions', 'Retweet', 'Handle'])

# create our training data from the tweets
print("Filtered train")
tr_data_df["Tweet_Text"] = process_map(parrallel_tokenize, [entry for entry in tr_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1) 
train = process_map(lemmatize_and_remove_stopwords, [entry for entry in tr_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1)
#train = pd.DataFrame(train)

#Using Unfiltered TFIDF
print("Unfiltered train")
unfiltered_data_df["Tweet_Text"] = process_map(parrallel_tokenize, [entry for entry in unfiltered_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1) 
unfiltered_vector = process_map(lemmatize_and_remove_stopwords, [entry for entry in unfiltered_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1)
#unfiltered_vector = pd.DataFrame(unfiltered_vector)

# create our test data from the tweets
print("Filtered test")
test_data_df["Tweet_Text"] = process_map(parrallel_tokenize, [entry for entry in test_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1) 
test = process_map(lemmatize_and_remove_stopwords, [entry for entry in test_data_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1)
#test = pd.DataFrame(test)

#Using Unfiltered TFIDF
print("Unfiltered test")
unfiltered_test_df["Tweet_Text"] = process_map(parrallel_tokenize, [entry for entry in unfiltered_test_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1) 
unfiltered_test = process_map(lemmatize_and_remove_stopwords, [entry for entry in unfiltered_test_df["Tweet_Text"].to_list()], max_workers=4, chunksize=1)
#unfiltered_test = pd.DataFrame(unfiltered_test)

# index all the sentiment labels
label = tr_data_df["Party"]



#max_features = 10000 --> builds a vocabulary that only considers the top max_fearures ordered by term frequency across the corpus
#If this is too big it makes the Conv+TFIDF imposible to train. 

#Pickle cleaned text TFIDF Vectorizer.
vectorizer = TfidfVectorizer(max_features=1000) #use_idf=True)
vectorizer = vectorizer.fit(train) 
with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

#Pickle Unfiltered text TFIDF Vectorizer. 
vectorizer_unfiltered = TfidfVectorizer(max_features=1000)
vectorizer_unfiltered.fit(unfiltered_vector)
with open('unfiltered_vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer_unfiltered, fin)

#Pickle Lemmatized Training vectors.
with open('trained_lemmas.pk', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Pickle Lemmatized Training vectors.
with open('unfiltered_trained_lemmas.pk', 'wb') as handle:
    pickle.dump(unfiltered_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Pickle Lemmatized test vectors.
with open('test_lemmas.pk', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Pickle Lemmatized test vectors.
with open('unfiltered_test_lemmas.pk', 'wb') as handle:
    pickle.dump(unfiltered_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
