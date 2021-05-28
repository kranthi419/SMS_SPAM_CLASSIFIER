# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:57:22 2021

@author: kaval
"""
import pandas as pd

messages = pd.read_csv('D:/SpamClassifier-master/smsspamcollection/SMSSpamCollection',sep='\t',names=['label','messages'])

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['messages'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english')) ]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'],drop_first=True).values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train,y_train)

y_pred = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_score = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
'''
import pickle
pickle_out = open("spam_classifier.pkl","wb")
pickle.dump(spam_detect_model, pickle_out)
pickle_out.close()'''