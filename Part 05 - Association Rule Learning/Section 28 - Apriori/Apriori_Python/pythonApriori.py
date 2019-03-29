#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:00:52 2019

@author: ts-fernando.takada
"""

# Apriori learning

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset and building product list
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None )
transactions = []

for i in range(0, 7502):
    transactions.append([str(dataset.values[i,j]) for j in range(0,19)])

# Training Apriori Model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the results
results = list(rules)