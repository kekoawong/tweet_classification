#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import graphviz

le = preprocessing.LabelEncoder()

# Get data, separate into features and labels
with open('../data/feature_vectors.obj', 'rb') as testing_file:
    data = pickle.load(testing_file)


data_df = pd.DataFrame.from_records(data, columns=["Party", "Tweet", "sentiment", "language", "dot_product", "vector", "is_retweet", "length_tweet", "num_hashtags", "num_mentions"])
data_df


# #### Encode the data

# In[18]:


data_encoded = data_df.copy()
data_encoded["Party"] = le.fit_transform(data_encoded["Party"])
data_encoded["Retweet"] = le.fit_transform(data_encoded["is_retweet"])
data_encoded["language"] = le.fit_transform(data_encoded["language"])
data_encoded


# #### Split into testing and training data

# In[19]:


train_df, test_df = train_test_split(data_encoded, test_size=0.2, random_state=42, shuffle=True)


# In[20]:


train_df


# In[21]:


test_df


# #### Get features of testing and training set

# In[22]:


# training set
x_train = train_df[["sentiment", "language", "dot_product", "is_retweet"]]
y_train = train_df[["Party"]]

# testing set
x_test = test_df[["sentiment", "language", "dot_product", "is_retweet"]]
y_test = test_df[["Party"]]


# #### Build ID3 tree using entropy

# In[26]:


clf_entropy = tree.DecisionTreeClassifier(criterion="entropy")
clf_entropy.fit(x_train, y_train)


# #### Display Tree

# In[30]:


dot_data_entropy = tree.export_graphviz(clf_entropy, out_file=None,
                                feature_names=["sentiment", "language", "dot_product", "is_retweet"],
                                class_names=["Democrat", "Republican"],
                                filled=True, rounded=True,
                                special_characters=True)  
graph_entropy = graphviz.Source(dot_data_entropy)
graph_entropy


# In[27]:


# Print Accuracy
y_pred_entropy = clf_entropy.predict(x_test)
print("ID3 Tree Results")
print("Predicted:", y_pred_entropy)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_entropy))


# #### Build tree using Gini

# In[28]:


clf_gini = tree.DecisionTreeClassifier()
clf_gini.fit(x_train, y_train)


# #### Display Gini Tree

# In[ ]:


dot_data_gini = tree.export_graphviz(clf_gini, out_file=None,
                                feature_names=["sentiment", "language", "dot_product", "is_retweet"],
                                class_names=["Democrat", "Republican"],
                                filled=True, rounded=True,
                                special_characters=True)  
graph_gini = graphviz.Source(dot_data_gini)


# In[29]:


# Print Accuracy
y_pred_gini = clf_gini.predict(x_test)
print("Gini Tree Results")
print("Predicted:", y_pred_gini)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_gini))


# In[32]:


# Random Forest
clf_rf = RandomForestClassifier(max_depth=7, min_samples_split=0.1, min_samples_leaf=0.05)
clf_rf.fit(x_train, y_train.values.ravel())
y_pred_rf = clf_rf.predict(x_test)
print("Predicted:", y_pred_rf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))


# In[ ]:




