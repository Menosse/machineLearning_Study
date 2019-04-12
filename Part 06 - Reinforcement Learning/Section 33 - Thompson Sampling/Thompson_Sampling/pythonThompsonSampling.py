#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 07:49:58 2019

@author: ts-fernando.takada
"""

# Implementing Thompson Sampling algorithm

# Clean variables and console
# %clear
# %reset -f
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N, d = dataset.shape[0],dataset.shape[1]
adSelected = []
numbersOfRewards_1 = [0] * d
numbersOfRewards_0 = [0] * d
totalReward = 0

for n in range (0, N):
    ad = 0
    maxRandom = 0
    for i in range(0, d):
        randomBeta = random.betavariate(numbersOfRewards_1[i] + 1, numbersOfRewards_0[i] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
    adSelected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbersOfRewards_1[ad] = numbersOfRewards_1[ad] + 1
    else:
        numbersOfRewards_0[ad] = numbersOfRewards_0[ad] + 1
    totalReward = totalReward + reward

# Visualising the results
plt.hist(adSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()