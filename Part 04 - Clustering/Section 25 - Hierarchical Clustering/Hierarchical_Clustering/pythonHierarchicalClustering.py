#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:17:57 2019

@author: ts-fernando.takada
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

consNumHc = 5

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3, 4]].values
# y = dataset.iloc[:,].values

# Using dendrongram to find out the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrongram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Fitting hierarchical clustering to the 'problem' dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
yHc = hc.fit_predict(x)

# Visualizing the clusters
plt.scatter(x[yHc == 0, 0], x[yHc == 0, 1], s = 100, c = 'red', label = 'Carefull')
plt.scatter(x[yHc == 1, 0], x[yHc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(x[yHc == 2, 0], x[yHc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(x[yHc == 3, 0], x[yHc == 3, 1], s = 100, c = 'black', label = 'Careless')
plt.scatter(x[yHc == 4, 0], x[yHc == 4, 1], s = 100, c = 'orange', label = 'Sensible')

plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()