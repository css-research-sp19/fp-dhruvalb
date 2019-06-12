'''
Program processes data in tokens
'''

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

nltk.download('stopwords')


#Aux Functions for Text Processing
def remove_numbers(string):
    return re.sub(r'\d+', '', string)

def basic_text_process(csv_file, traveltype=None):
    # Convert to lower case
    # Remove punctuation
    # Remove numbers

    data = pd.read_csv('full_data.csv')
    if traveltype:
        data = data[data['Traveller Type'] == traveltype]

    data['Text'] = data['Text'].str.lower()
    data['Text'] = data['Text'].str.replace('[^\w\s]','')
    data['Text'] = data['Text'].apply(remove_numbers)

    return data

# AUX Function to process Tokens
# Remove Stop Words
def remove_stop(words):
    stop_words = stopwords.words('english')
    domain_words = set(['san', 'francisco', 'green', 'tortoise', 'hostel', 'orange', 'village'])
    rem_words = []
    for word in words:
        if word not in stop_words and word not in domain_words:
            rem_words.append(word)
    return rem_words

# Apply Porter Stem
def apply_stem(words):
    porter = PorterStemmer()
    ret_words = []

    for word in words:
        ret_words.append(porter.stem(word))

    return ret_words

def count_tokens(text_dataframe):
    '''
    Method to create tokens for reviews
    input df['text'].values
    '''
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_dataframe)
    full_features = vectorizer.get_feature_names()
    no_stop_features = remove_stop(full_features)
    #features = apply_stem(no_stop_features)
    features = no_stop_features

    return features

def tfidf_tokens(text_dataframe):
    '''
    Method to create tokens for reviews
    input df['text'].values
    '''
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_dataframe)
    # full_features = vectorizer.get_feature_names()
    # no_stop_features = remove_stop(full_features)
    # features = apply_stem(no_stop_features)

    return X

