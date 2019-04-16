# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Clear console and variable explorer
# %clear
# %reset -f

# Import the libraries
import pandas as pd
import re
# 2 Steps Below is to download stopwords libraries, it is necessary only once.
import nltk
import numpy as np
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Importing the dataset, including the delimiter of a TAB and IGNORING the quoting
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
D = dataset.shape[0]

# Cleaning the text
corpus = []
for i in range (0, D):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create the Bag of words model - Create Sparse matrix through tokenization
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting dataset into trainingset and testset
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting classifier using NAIVE BAYES model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xTrain, yTrain)

# Predictin test dataset results
yPred = classifier.predict(xTest)

# Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)