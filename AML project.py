# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:06:48 2018

@author: 18123
"""
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt


#Importing the train and test files into Data Frames.
train_file = pd.read_csv("train.tsv", sep="\t")
test_file = pd.read_csv("test.tsv", sep="\t")



#Checking for imbalance of Labels
plot_imbal_labels = train_file.groupby(["Sentiment"]).size()
plot_imbal_labels = plot_imbal_labels / plot_imbal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_imbal_labels.keys(), plot_imbal_labels.values);


"""
First step is balancing the imbalanced data the histogram gives the information
about the different sentiments. As we can see the data is imbalanced with neutral data 
being more in our dataset compared to other it is taking almost 50 percent of the 

"""

# Separate majority and minority classes
train_majority = train_file[train_file.Sentiment==2]
train_minority = train_file[train_file.Sentiment!=2]


# Downsample majority class

train_majority_downsampled = resample(train_majority, replace=False, n_samples=27273, random_state=123) 

## Combine minority class with downsampled majority class
train_downsampled = pd.concat([train_majority_downsampled, train_minority])
plot_train_downsampled = train_downsampled.groupby(["Sentiment"]).size()
plot_train_downsampled  = plot_train_downsampled  / plot_train_downsampled.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_train_downsampled.keys(), plot_train_downsampled.values)


#Data Cleaning of train set
def preprocess(given_review):
    review = given_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))


#Applying the preprocess function on the Phrase coloumn to clean the data
train_downsampled['Cleaned_Phrase'] = train_downsampled["Phrase"].apply(lambda x :preprocess(x))



#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)













