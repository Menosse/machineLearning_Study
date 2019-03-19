#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:31:17 2019

@author: ts-fernando.takada
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting train set and test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain, xTest = sc.fit_transform(xTrain), sc.transform(xTest)

# Fitting classifier to training set
# Create new SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(xTrain, yTrain) 

# Prediction test set results
yPred = classifier.predict(xTest)

# Making confusion matrixs
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

# Plotting the results
from matplotlib.colors import ListedColormap
xSet, ySet = xTest, yTest
x1, x2 = np.meshgrid(np.arange(start = xSet[:, 0].min() - 1, stop = xSet[:, 0].max() + 1, step = 0.1),
                     np.arange(start = xSet[:, 1].min() - 1, stop = xSet[:, 1].max() + 1, step = 0.1))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x1.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(xSet[ySet == j, 0], xSet[ySet == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
