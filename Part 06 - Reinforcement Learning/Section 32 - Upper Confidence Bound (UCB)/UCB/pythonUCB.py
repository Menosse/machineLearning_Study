#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:00:52 2019

@author: ts-fernando.takada
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implement UCB algorithm
import math
N , d = int(dataset.shape[0]),int(dataset.shape[1])
adSelected = []
numbersOfSelections = [0] * d
sumsOfReward = [0] * d
totalReward = 0
for n in range (0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if (numbersOfSelections[i] > 0):
            average_reward = sumsOfReward[i] / numbersOfSelections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/numbersOfSelections[i])
            upperBound = average_reward + delta_i
        else:
            upperBound = 1e400
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
    adSelected.append(ad)
    numbersOfSelections[ad] = numbersOfSelections[ad] + 1
    reward = dataset.values[n, ad]
    sumsOfReward[ad] = sumsOfReward[ad] + reward
    totalReward = totalReward + reward

# Visualising the results
plt.hist(adSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()