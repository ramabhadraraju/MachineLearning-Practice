# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:58:40 2017

@author: Raju garu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('E:/Udemy MachineLearning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#FIt linear regression
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X,y)
#fit polynomial
from sklearn.preprocessing import PolynomialFeatures
polyreg= PolynomialFeatures(degree=2)
x_poly= polyreg.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)

#Visualizing Linear results
plt.scatter(X,y,color='red')
plt.plot(X,lr.predict(X),color='blue')
plt.title('truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#Visualizing Polynomial regression results

plt.scatter(X,y,color='red')
plt.plot(X,lr2.predict(x_poly),color='blue')
plt.title('truth or bluff(PolynomialRegression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()






