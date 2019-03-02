# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 07:35:09 2019

@author: ts-fernando.takada
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# IMPORTING DATASET
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# TAKING CARE OF MISSING DATA
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode labels
labelEncoderX = LabelEncoder()
x[:,0] = labelEncoderX.fit_transform(x[:, 0])
XoneHotEncoder = OneHotEncoder(categorical_features= [0])
x = XoneHotEncoder.fit_transform(x).toarray()

# Encode yes/no variable
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Data Scaling
from sklearn.preprocessing import StandardScaler

scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest)

#xTrain = scX.fit_transform(xTrain)
#xTest = scX.transform(xTest)


