#!/usr/bin/env python
# coding: utf-8

import pickle
def save_object(file_name, ob):
    with open(file_name, 'wb') as outfile:
        pickle.dump(ob, outfile)

# In[17]:

import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graphviz

graph_file_entropy = '../visualizations/entropy_graph.png'
graph_file_gini = '../visualizations/gini_graph.png'
graph_file_rf = '../visualizations/random_forest_graph.png'

le = preprocessing.LabelEncoder()

print('Loading')
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

print('Making graph')
dot_data_entropy = tree.export_graphviz(clf_entropy, out_file=None,
                                feature_names=["sentiment", "language", "dot_product", "is_retweet"],
                                class_names=["Democrat", "Republican"],
                                filled=True, rounded=True,
                                special_characters=True)  
graph_entropy = graphviz.Source(dot_data_entropy)
save_object(graph_file_entropy, graph_entropy)


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
print(graph_gini)
save_object(graph_file_gini, graph_gini)

# In[29]:


# Print Accuracy
y_pred_gini = clf_gini.predict(x_test)
print("Gini Tree Results")
print("Predicted:", y_pred_gini)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_gini))


# In[32]:


# Random Forest
'''clf_rf = RandomForestClassifier(max_depth=7, min_samples_split=0.1, min_samples_leaf=0.05)
clf_rf.fit(x_train, y_train.values.ravel())
y_pred_rf = clf_rf.predict(x_test)
print("Predicted:", y_pred_rf)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))

dot_data_rf = tree.export_graphviz(clf_rf, out_file=None,
                                feature_names=["sentiment", "language", "dot_product", "is_retweet"],
                                class_names=["Democrat", "Republican"],
                                filled=True, rounded=True,
                                special_characters=True)  
graph_rf = graphviz.Source(dot_data_rf)
save_object(graph_file_rf, graph_rf)'''


# In[ ]:




