# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:19:32 2018

@author: 18123
"""
import pandas as pd
train_downsamp_preproces = pd.read_csv("train_proj.tsv", sep="\t")
train_downsamp_preproces = train_downsamp_preproces.dropna()




#Splitting into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_downsamp_preproces['Cleaned_Phrase'],train_downsamp_preproces['Sentiment'], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

X_train_svm = X_train.copy()
y_train_svm = y_train.copy()
X_test_svm = X_test.copy()
y_test_svm = y_test.copy()
from sklearn.metrics import accuracy_score
#Count Vectorizing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X_train = cv.fit_transform(X_train).toarray()
X_test= cv.fit_transform(X_test).toarray()

#Implementation of Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
score = accuracy_score(y_test,y_test_pred)

##################################################################################
###################COPIED####################


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

import numpy as np




svc = LinearSVC(
    C=1.0,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=1000,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05, 
    verbose=0
)

tfidf = CountVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 2),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64
)

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svc', svc)
])






pipeline.fit(X_train_svm, y_train_svm)
train_score = pipeline.score(X_train_svm, y_train_svm)
test_score = pipeline.score(X_test_svm, y_test_svm)
print("Train = {}, Test = {}".format(train_score, test_score))
