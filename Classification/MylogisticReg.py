# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:13:42 2017

@author: Raju garu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('E:\\Udemy MachineLearning\\Machine Learning A-Z Template Folder\\Part 3 - Classification\\Section 14 - Logistic Regression\\Logistic_Regression\\Social_Network_Ads.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

#Dummy encoding for gender column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#dummyvariable trap
X= X[:, 1:]

#train test Split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fit Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#Predicting Test set
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Visualizing the results










