# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 07:35:09 2019

@author: ts-fernando.takada
"""
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest) '''
