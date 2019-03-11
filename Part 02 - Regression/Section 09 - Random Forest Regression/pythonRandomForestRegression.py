#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:03:02 2019

@author: ts-fernando.takada
"""

# Random Forest Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
forestDataset = pd.read_csv('Position_Salaries.csv')
x = forestDataset.iloc[:, 1:2].values
y = forestDataset.iloc[:, 2].values

# Fitting regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
forestRegressor = RandomForestRegressor(n_estimators=500, random_state = 0)
forestRegressor.fit(x,y)

# Predicting new result
yPred = forestRegressor.predict(np.array([[6.5]]))

# Create random forest chart
xGrid = np.arange(min(x), max(x), 0.01)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(x,y, color = 'red')
plt.plot(xGrid, forestRegressor.predict(xGrid), color = 'blue')
plt.title('Truth or bluff (Random Forest)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()￼