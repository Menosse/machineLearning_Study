# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 07:53:16 2019

@author: ts-fernando.takada
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Spliting Dataset to into train set and test set
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(x,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
xPoly = polyReg.fit_transform(x)
linReg2 = LinearRegression()
linReg2.fit(xPoly, y)

# Visualize the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x,linReg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression results
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(xGrid,linReg2.predict(polyReg.fit_transform(xGrid)), color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with Linear Regression
linReg.predict(6.5)

# Predict a new result with Polynomial regression
linReg2.predict(polyReg.fit_transform(6.5))