# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:05:35 2019

@author: Hello
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

email_data = pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Naive Bayes\\Datasets\\sms_raw_NB.csv",encoding = "ISO-8859-1")

##Cleaning data
import re
stop_words = []
with open("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Naive Bayes\\Datasets\\stop.txt") as f:
    stop_words = f.read()
    
##As Stopwards are in a single string, lets convert into list of single words
    
stop_words = stop_words.split("\n")

##Defining a custom function for cleaning the data
def cleaningdata (i):
    i= re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w= []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))
    
##Applying the custome function to email data text column

email_data["text"]= email_data["text"].apply(cleaningdata)

##Removing the empty rows if any generated
email_data.shape
email_data = email_data.loc[email_data.text != " ",:]
##There are no empty spaces

##Creating a matrix of token counts for the entire text document
def split_if_words(i):
    return [word for word in i.split(" ")]

predictors = email_data.iloc[:,1]
target = email_data.iloc[:,0]
#Splitting the data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(predictors, target, test_size = 0.3, stratify = target)

##Convert email text into word count matric i.e bag of words
email_bow = CountVectorizer(analyzer = split_if_words).fit(email_data["text"])

##For all the emails doing the transformation

all_emails_matrix = email_bow.transform(email_data["text"])
all_emails_matrix.shape
#(5559, 6661)

##For training data
train_emails_matrix = email_bow.transform(x_train)
train_emails_matrix.shape
#(3891, 6661)

##For test data
test_emails_matrix = email_bow.transform(x_test)
test_emails_matrix.shape
##(1668, 6661)

##Building the model without doing the TFIDF###
##Preparing the Naive Bayes model

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

##Building the Multinomial naive bayes model

classifier_nb = MB()
classifier_nb.fit(train_emails_matrix,y_train)
train_pred_nb =classifier_nb.predict(train_emails_matrix) 
accuracy_nb = np.mean(train_pred_nb==y_train)
##98.8%
pd.crosstab(train_pred_nb, y_train)

##predicting on test data
test_pred_nb = classifier_nb.predict(test_emails_matrix)
accuracy_test_nb = np.mean(test_pred_nb == y_test )
##96.82%
pd.crosstab(test_pred_nb,y_test)

##Building Gaussian model

classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),y_train.values)
train_pred_gb = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_gb = np.mean(train_pred_gb == y_train)
##92%
pd.crosstab(train_pred_gb,y_train)

##predicting on test data
test_pred_gb = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_gb = np.mean(test_pred_gb == y_test)
##85%
pd.crosstab(test_pred_gb, y_test)

###Building with TFIDF transformation
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

##Preparing Tfidf for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape
##(3891, 6661)
##Preparing Tidf f0or test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape
##(1668, 6661)

###Building Multinomial Naive Bayes model
classifer_mb_tfidf = MB()
classifer_mb_tfidf.fit(train_tfidf,y_train)
train_predmb_tfidf = classifer_mb_tfidf.predict(train_tfidf)
accuracy_mb_tfidf = np.mean(train_predmb_tfidf == y_train)
##96.58%
pd.crosstab(train_predmb_tfidf, y_train)

test_predmb_tfidf = classifer_mb_tfidf.predict(test_tfidf)
accuracy_testmb_tfidf = np.mean(test_predmb_tfidf == y_test)
##96%

##Building gaussiam naive bayes model
classifier_gb_tfidf = GB()
classifier_gb_tfidf.fit(train_tfidf.toarray(),y_train.values)
train_predgb_tfidf = classifier_gb_tfidf.predict(train_tfidf.toarray())
accuracy_gb_tfidf = np.mean(train_predgb_tfidf == y_train)
##92%
pd.crosstab(train_predgb_tfidf,y_train)

test_predgb_tfidf = classifier_gb_tfidf.predict(test_tfidf.toarray())
accuracy_testgb_tfidf = np.mean(test_predgb_tfidf == y_test)
##85%
pd.crosstab(test_predgb_tfidf,y_test)
