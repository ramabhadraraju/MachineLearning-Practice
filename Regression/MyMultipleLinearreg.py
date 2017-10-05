# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:07:23 2017

@author: Raju garu
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('E:/Udemy MachineLearning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#Removing the dummyvariable trap
X=X[:,1:]
#test train split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting MLR
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,Y_train)

#Predicting test results
y_pred=lr.predict(X_test)

#using backward elimination using stats models
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#removing dummyvariable x2 due to large p value
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing 2nd dummy large pvalue
X_opt=X[:,[0,3,4,5]]
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#Removing admin
X_opt=X[:,[0,3,5]]
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing mark
X_opt=X[:,[0,3]]
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()







