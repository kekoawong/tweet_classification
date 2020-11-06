import json
import pickle
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
import math
import numpy

stemmer_en = SnowballStemmer("english")
stemmer_sp = SnowballStemmer("spanish")

output_file_name = '../data/sentiment_and_language.obj'

# load files
with open('../data/training_data.obj', 'rb') as training_file:
    training_data = pickle.load(training_file)
with open('../data/sentiment_words_en.json') as sent_words:
    sentiment_words_en = json.load(sent_words)
with open('../data/sentiment_words_es.json') as sent_words:
    sentiment_words_sp = json.load(sent_words)
with open('../data/gloVe_vectors.obj', 'rb') as input_file:
    glove_words = pickle.load(input_file)

tweets = []
for index, tweet in enumerate(training_data['tweets']):
    
    # set variables
    scores = []
    tweet_words = tweet[1].lower()
    tweet_vec = []
    avg_word_vec = []
    dot_product = 0
    
    try:
        lang = detect(tweet_words)
    except:
        print('LangDetectException')
        print(tweet_words)
        continue

    for word in tweet_words.split():
        # if spanish
        if lang == 'es':
            root_word = stemmer_sp.stem(word)
            if root_word in sentiment_words_sp:
                scores.append(sentiment_words_sp[root_word])
        elif lang == 'en':
            root_word = stemmer_en.stem(word)
            if root_word in sentiment_words_en:
                scores.append(sentiment_words_en[root_word])


        if word in glove_words or root_word in glove_words:
            if word not in glove_words:
                word = root_word
            # compute tweet dot from combining all word vectors
            if len(tweet_vec) == 0:
                tweet_vec = glove_words[word]
            else:
                dot_product = numpy.dot(tweet_vec, glove_words[word])
        
            # compute tweet_vector from average of word vectors
            if len(avg_word_vec) == 0:
                avg_word_vec = glove_words[word]
            else:
                avg_word_vec = [ (avg_word_vec[i] + glove_words[word][i]) / 2 for i in range(len(avg_word_vec)) ]

    # check if words for score
    if len(scores) == 0:
        continue
    # Sorted from greatest to least absolute value
    sorted_abs_list = sorted(map(abs, scores), reverse=True)
    # take the sentiment rough estimate, divided by 4 for better results
    nums = math.ceil(len(scores) / 4)
    computed_score = sum(sorted_abs_list[:nums]) / nums

    # store object as [Party, Tweet, sentiment, language, dot_product, vector, is_retweet, length_tweet, num_hashtags, num_mentions]
    tweets.append([ tweet[0], tweet[6], computed_score, lang, dot_product, avg_word_vec, tweet[4], len(tweet[6]), len(tweet[2]), len(tweet[3]) ])
    
    # print loading output
    if index % 5000 == 0:
        print(f'{[tweet[6], computed_score, dot_product, avg_word_vec]} with index {index}')
    
# write to pickle file
print('Writing to file')
with open(output_file_name, 'wb') as out_file:
    pickle.dump(tweets, out_file)
print('Done')