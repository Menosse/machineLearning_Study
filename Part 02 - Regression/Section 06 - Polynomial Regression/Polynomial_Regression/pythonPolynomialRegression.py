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
x = dataset.iloc[:,-1].values
y = dataset.iloc[:,3].values

# Spliting Dataset to into train set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrian, yTest