# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:19:32 2019

@author: ts-fernando.takada
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing DataSet
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# ENCODE Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
x[:, 3] = labelEncoderX.fit_transform(x[:, 3])
oneHotEncoderX = OneHotEncoder(categorical_features=[3])
x = oneHotEncoderX.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap
x = x[:, 1:] # It is not necessary to do it manually, py Library does it to me

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size = 0.2, random_state = 0)

# Feature Scaling - It is not necessary for multiple linear regression, the library does it to me

# Fitting Multiple Linear Regressio to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain,yTrain)

# Predicting the test set results
yPred = regressor.predict(xTest)

# Building the optimal model using Backward Elimination

# 3 fisrt models does are not good enough because the P-value and Adj-Rsquared are not
# in the best fit to the model
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1 )
xOpt =  x[:,[0, 1, 2, 3, 4, 5]]
regressorOls = sm.OLS(endog= y, exog= xOpt).fit()
regressorOls.summary()

xOpt =  x[:,[0, 1, 3, 4, 5]]
regressorOls = sm.OLS(endog= y, exog= xOpt).fit()
regressorOls.summary()

xOpt =  x[:,[0, 3, 4, 5]]
regressorOls = sm.OLS(endog= y, exog= xOpt).fit()
regressorOls.summary()

# This is the best model because the Adj-Rsquared and P-value has the higher importance
# It means that for this dataset R.D. Spend and Marketing Spend are the most important
# Variables to startup growth
xOpt =  x[:,[0, 3, 5]]
regressorOls = sm.OLS(endog= y, exog= xOpt).fit()
regressorOls.summary()

# This is not a good adjust because the Adj-Rsquared is lower
xOpt =  x[:,[0, 3]]
regressorOls = sm.OLS(endog= y, exog= xOpt).fit()
regressorOls.summary()
