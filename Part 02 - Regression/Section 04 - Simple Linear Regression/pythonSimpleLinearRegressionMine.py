# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 07:35:09 2019

@author: ts-fernando.takada
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest) '''


# Fitting Simple linear Regression model to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

# Predicting the testing set results
yPred = regressor.predict(xTest)

# Visualising the Training set results
plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set results
plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()