import logging
import pandas as pd
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#################################################################
#################################################################

#upload data!
train_data = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\train01.csv')
test_data = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\test01.csv')

#print(train_data.head(10))

##setup the data
my_tags = [ 'FALSE', 'TRUE', 'half-true', 'mostly-true', 'barely-true', 'pants-fire']
plt.figure(figsize=(10,4))
train_data.LABEL.value_counts().plot(kind='bar')

X = train_data.TEXT
y = train_data.LABEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


###################################################
###################################################

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

#what is pipeline?
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)


from sklearn.metrics import classification_report
y_pred = nb.predict(X_test) ##wats this?

print("---------------------the classification metric for NB:----------------")
print('accuracy of Naive-Bayes %s ' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))

##################################################################
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print("---------the classication metric for SVM-------------------------")
print('accuracy (SVM) %s ' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))


################################################################
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print("---------------the classification metric for Logistic Regression---------------")
print('accuracy (Logistic regression) %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))

###################################################################
from sklearn.ensemble import RandomForestClassifier

rfc = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(n_estimators = 200, max_depth=3, random_state=0)), #this will change
               ])
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("---------------the classification metric for Random Forest Classifier---------------")
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))




