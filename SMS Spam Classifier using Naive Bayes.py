#!/usr/bin/env python
# coding: utf-8

# # Importing the Dataset

# In[1]:


import pandas as pd


# In[3]:


messages = pd.read_csv(r"C:\Users\tamba\Downloads\SMSSpamCollection", sep='\t',names=["label", "message"])


# # Data cleaning and preprocessing

# In[4]:


import re #Reguler Expression
import nltk
nltk.download('stopwords')


# In[5]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) #Remove all character except a-z and A-Z(substitute with blank' '.)
    review = review.lower() #Lower all sentences
    review = review.split() #Getting list of word
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# # Creating the Bag of Words model

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()


# In[7]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# # Train Test Split

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Training model using Naive bayes classifier

# In[9]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test) 


# In[13]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_m=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print(confusion_m)

