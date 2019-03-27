#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:35:14 2019

@author: ts-fernando.takada
"""
# %reset -f
# %clear

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

# Creating K-Means module using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
   
# Plotting the amount of clusters using elbow method
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means to the mall dataset with the correct amount of clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
yKmeans = kmeans.fit_predict(x)

# Visualizing the clusters
plt.scatter(x[yKmeans == 0, 0], x[yKmeans == 0, 1], s = 100, c = 'red', label = 'Carefull')
plt.scatter(x[yKmeans == 1, 0], x[yKmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(x[yKmeans == 2, 0], x[yKmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(x[yKmeans == 3, 0], x[yKmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(x[yKmeans == 4, 0], x[yKmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300, c = 'yellow', label = 'centroid')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()