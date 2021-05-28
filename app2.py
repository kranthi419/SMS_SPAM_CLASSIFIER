# -*- coding: utf-8 -*-
"""
@author: K. kranthi kumar

"""

# -*- coding: utf-8 -*-


import streamlit as st 
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

ps = PorterStemmer()


from PIL import Image

classes = ['ham','spam']

@st.cache
def train():
    # This function will only be run the first time it's called
    messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','messages'])
    corpus = []
    cv = CountVectorizer(max_features=2500)    
    for i in range(0,len(messages)):
        review = re.sub('[^a-zA-Z]',' ',messages['messages'][i])
        review = review.lower()
        review = review.split()
                
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english')) ]
        review = ' '.join(review)
        corpus.append(review)
            
    
    x = cv.fit_transform(corpus).toarray()
    y = pd.get_dummies(messages['label'],drop_first=True).values
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    from sklearn.naive_bayes import MultinomialNB
    spam_detect_model = MultinomialNB().fit(x_train,y_train)
    return cv,spam_detect_model 

def predict_note_authentication(sentences,cv,spam_detect_model):

    sentences = re.sub('[^a-zA-Z]',' ',sentences)
    sentences = sentences.lower()
    sentences = sentences.split()
    sentences = [ps.stem(word) for word in sentences if word not in set(stopwords.words('english')) ]
    sentences= ' '.join(sentences)
    x2 = cv.transform([sentences]).toarray()
   
    prediction=spam_detect_model.predict(x2)
    print(classes[int(prediction)])
    return classes[int(prediction)]


def main():
    st.title("Spam classifier")
    html_temp = """
    <div style="background-color:darkslategray;padding:20px">
    <h2 style="color:white;text-align:center;">Streamlit Spam classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sentence = st.text_input("sentence","Type Here")
    result=""
    if st.button("Predict"):
        cv,spam_detect_model = train()
        result=predict_note_authentication(sentence,cv,spam_detect_model)
    st.success('the message is ' + result )
    if st.button("About"):
        st.text("Spam_classifier")
        st.text("Built with Streamlit")
   

if __name__=='__main__':
    main()
    
    
    