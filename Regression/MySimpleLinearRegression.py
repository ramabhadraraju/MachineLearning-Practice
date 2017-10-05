# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#test train split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#import linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)

#plot the data for training
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train, lr.predict(X_train),color='blue')
plt.plot(X_test,y_pred)
plt.title('Salary VS Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

#Plot for test set
plt.scatter(X_test,Y_test, color='green')
plt.plot(X_train, lr.predict(X_train),color='blue')
plt.title('Salary VS Experience(Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()














