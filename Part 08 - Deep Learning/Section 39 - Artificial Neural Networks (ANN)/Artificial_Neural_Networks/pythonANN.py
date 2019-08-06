#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 07:29:23 2019

@author: ts-fernando.a.takada
"""

# installing Theano 
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Install Tensorflow
# Google it

# Install Keras
# pip install --upgrade keras

''' Data Preprocessing'''

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

''' START DATA PRE PROCESSING!! '''

# Encode categorical data
# Import library Encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode Country categorical variable
lbX = LabelEncoder()
x[:,1] = lbX.fit_transform(x[:,1])

# encode Gender categocircal variable
lbX_1 = LabelEncoder()
x[:,2] = lbX_1.fit_transform(x[:,2])

# Create dummy variables for countries encoding
ohe_x1 = OneHotEncoder(categorical_features=[1])
x = ohe_x1.fit_transform(x).toarray()

# Avoid dummy variable TRAP!!
x = x[:, 1:]

# Splitting the data into training and test datasets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling for DEEP LEARNING? YES!!!

# Import feature scaling library
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)

'''FINISH DATA PRE PROCESSING!! '''

''' START TO FIT ARTIFICIAL NEURAL NETWORK ON TRAINING SET!! (ANN)'''

# Import KERAS library and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(xTrain, yTrain, batch_size = 10, nb_epoch = 100)

''' FINISH TO FIT ARTIFICIAL NEURAL NETWORK ON TRAINING SET!! '''

''' START TO TEST THE ANN ON TEST SET '''

# Predicting the test set result 

yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)
# Making confusion matrix

# Import confusion matrix library
from sklearn.metrics import confusion_matrix

# Make confusion matrix
cm = confusion_matrix(yTest, yPred)

''' FINISH TO TEST THE ANN ON TEST SET '''