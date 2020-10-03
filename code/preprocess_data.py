# Same preprocessing data, converted to a script

import pandas as pd
import pickle
import random
import string
import preprocessor as preprocess

final_data = {'tweets': []}

rep_tweets_df = pd.read_csv("../data/raw/ExtractedTweets.csv")
rep_tweets_df

# Process first dataset
for line in rep_tweets_df.iterrows():
    
    # update variables
    tweet = []
    tweet_content = line[1]['Tweet']
    if not tweet_content:
        continue
    retweet = tweet_content.startswith('RT')
    handle = line[1]['Handle']
    party = line[1]['Party']
    hashtags = []
    mentions = []
    
    # get hashtags and mentions
    parsed_tweet = preprocess.parse(tweet_content)
    if parsed_tweet.hashtags:
        hashtags = [ht.match for ht in parsed_tweet.hashtags]
    if parsed_tweet.mentions:
        mentions = [m.match for m in parsed_tweet.mentions]
    
    # clean tweet for just words and make dictionary object
    words = preprocess.clean(tweet_content)
    # check if more than one letter
    if not words:
        continue
    tweet = [party, words.translate(str.maketrans('', '', string.punctuation)), hashtags, mentions, retweet, handle]
    final_data['tweets'].append(tweet)

print('Done with dataset 1')

candidate_tweets_df = pd.read_csv("../data/raw/tweets.csv")
candidate_tweets_df

for line in candidate_tweets_df.iterrows():

    # update variables
    tweet = []
    tweet_content = line[1]['text']
    if not tweet_content:
        continue
    retweet = line[1]['is_retweet']
    handle = line[1]['handle']
    party = 'Democrat' if handle == 'HillaryClinton' else 'Republican'
    hashtags = []
    mentions = []
    
    # get hashtags and mentions
    parsed_tweet = preprocess.parse(tweet_content)
    if parsed_tweet.hashtags:
        hashtags = [ht.match for ht in parsed_tweet.hashtags]
    if parsed_tweet.mentions:
        mentions = [m.match for m in parsed_tweet.mentions]
    
    # clean tweet for just words and make dictionary object
    words = preprocess.clean(tweet_content)
    # check if more than one letter
    if not words:
        continue
        
    # get retweet for some tweets that are surrounded by quotes
    if words.startswith('/":"'):
        retweet = True
    tweet = [party, words.translate(str.maketrans('', '', string.punctuation)), hashtags, mentions, retweet, handle]
    final_data['tweets'].append(tweet)

print('Done with dataset 2')

# define percent to be testing data
percent_testing = 0.2
testing_data = {'tweets': []}

length = len(final_data['tweets'])
testing_amount = int( percent_testing * length )

for t in range(0, testing_amount):
    length = len(final_data['tweets'])
    n = random.randint(0,length-1)
    tw = final_data['tweets'].pop(n)
    testing_data['tweets'].append(tw)
print('Done seperating')

with open('../data/testing_data.json', 'wb') as testing_file:
    pickle.dump(testing_data, testing_file)

with open('../data/training_data.json', 'wb') as training_file:
    pickle.dump(final_data, training_file)
    
print('Complete')