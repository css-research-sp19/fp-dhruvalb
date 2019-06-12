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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')

#Basic Workflow: 
'''
To get model accuracy: 
create dataframe of DTM and output classification. 
Use dt_model or rf_model to create model and output cv accuracies df
Create cv plot using create_cv_plot
'''


#Aux Functions for Text Processing
def remove_numbers(string):
    return re.sub(r'\d+', '', string)

def basic_text_process(csv_file):
    # Convert to lower case
    # Remove punctuation
    # Remove numbers
    data = pd.read_csv('full_data.csv')
    data['Text'] = data['Text'].str.lower()
    data['Text'] = data['Text'].str.replace('[^\w\s]','')
    data['Text'] = data['Text'].apply(remove_numbers)

    return data

# AUX Function to process Tokens
# Remove Stop Words
def remove_stop(words):
    '''
    Remove stop words in english language and some specific words
    '''
    stop_words = stopwords.words('english')
    domain_words = set(['san', 'francisco', 'green', 'tortoise', 'hostel', 'orange', 'village'])
    domain_words.update(['ive', 'would', 'hi', 'center', 'city', 'really', 'hostels'])
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

def create_tokens(text_dataframe):
    '''
    Method to create tokens for reviews
    input df['text'].values
    '''
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, \
        stop_words='english', strip_accents= 'ascii')

    X = vectorizer.fit_transform(text_dataframe)

    features = vectorizer.get_feature_names()

    # no_stop_features = remove_stop(full_features)
    # features = apply_stem(no_stop_features)

    return X, features


def convert(string):
    if string == 'Solo':
        return 1
    else:
        return 0

def create_matrix(csv_file):
    data = basic_text_process(csv_file)
    token = create_tokens(data['Text'].values)

    df = pd.DataFrame(token[0].toarray(), columns=token[1])
    df['Type'] = data['Traveller Type']

    df['Type'] = df['Type'].apply(convert)
    
    return df


# Create Model
def dt_model(df):
    '''
    Given processed dataframe with word matrix and output:
    - creates train and test split data 
    - searches for optimal paramenters
    - creates decision tree model
    - prints the result for time, cross validation and test accuracy
    - outputs a dataframe with trials and cv accuracy. 
    '''

    X = df.loc[:, df.columns != 'Type']

    X_train, X_test, y_train, y_test = train_test_split(X, df['Type'], test_size=0.9, random_state=0)

    max_dept = np.arange(2, 9)
    fraction = np.arange(0.0, 0.5, 0.05)

    tree_param_grid = {'random_state': [1],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': max_dept,
                       'min_weight_fraction_leaf': fraction}

    tic = time.time()
    tree = DecisionTreeClassifier(random_state=0)
    tree_gs = RandomizedSearchCV(tree, tree_param_grid, cv=10,
                                 scoring='accuracy', n_jobs=-1,
                                 iid=False, return_train_score=True, random_state=0)
    tree_gs.fit(X_train.values, y_train.values)
    toc = time.time()

    print("Time elapsed: {:.3f} seconds".format(toc - tic))
    print('Best hyperparameter setting for Decision Tree is {}'.format(tree_gs.best_params_))
    print('5-fold accuracy score is {:.3f}'.format(tree_gs.best_score_))

    y_test_pred = tree_gs.predict(X_test)
    test_accu = metrics.accuracy_score(y_test, y_test_pred)
    print("Test set accuracy rate is {:.3f}".format(test_accu))

    tree.fit(X_train.values, y_train.values)
    importances = tree.feature_importances_
    print(importances)

    #Create Cross Validation DF
    valid = np.arange(1,11)
    all_accuracies = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10)  
    print(all_accuracies)
    df_acc = pd.DataFrame({'Trial':valid, 'Accuracy':all_accuracies})


    return df_acc

def rf_model(df):
    '''
    Given processed dataframe with word matrix and output:
    - creates train and test split data 
    - searches for optimal paramenters
    - creates random forest model
    - prints the result for time, cross validation and test accuracy
    - outputs a dataframe with trials and cv accuracy. 
    '''
    X = df.loc[:, df.columns != 'Type']

    X_train, X_test, y_train, y_test = train_test_split(X, df['Type'], test_size=0.9, random_state=0)

    max_dept = np.arange(2, 9)
    fraction = np.arange(0.0, 0.5, 0.05)


    rf_param_grid = {'n_estimators': [10, 50, 100, 200, 300, 400, 500, 1000],
                  'random_state': [0],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': max_dept,
                  'min_weight_fraction_leaf': fraction,
                  'n_jobs': [-1],
                  'bootstrap': [True, False]}


    tic = time.time()
    rf = RandomForestClassifier(random_state=0)
    rf_gs = RandomizedSearchCV(rf, rf_param_grid, cv=5,
                                 scoring='accuracy', n_jobs=-1,
                                 iid=False, return_train_score=True, random_state=0)
    rf_gs.fit(X_train.values, y_train.values)
    toc = time.time()

    print("Time elapsed: {:.3f} seconds".format(toc - tic))
    print('Best hyperparameter setting for Decision Tree is {}'.format(rf_gs.best_params_))
    print('5-fold accuracy score is {:.3f}'.format(rf_gs.best_score_))

    y_test_pred = rf_gs.predict(X_test)

    test_accu = metrics.accuracy_score(y_test, y_test_pred)
    print("Test set accuracy rate is {:.3f}".format(test_accu))

    #Create Cross Validation DF
    valid = np.arange(1,6)
    all_accuracies = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=5)
    df_acc = pd.DataFrame({'Trial':valid, 'Accuracy':all_accuracies})


    return df_acc


#Plots:

def create_cv_plot(df):
    '''
    Use this to create a plots for cross validation, given dataframe of trials and accuracy
    '''
    ax = sns.lineplot(x="Trial", y="Accuracy", data=df, palette="PiYG")
    ax.set_title('Random Forest: Accuracy Plotted over Cross Validation Trials')
    #ax.set_color('darkmagenta')
    plt.show()
