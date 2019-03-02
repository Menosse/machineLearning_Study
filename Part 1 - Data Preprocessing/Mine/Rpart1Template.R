# DATA PREPROCESSING

# IMPORT DATA SET
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]


# #Splitting the dataset into training set and test set
# install.packages('caTools')

# library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# # Feature Scaling
trainingSet[, 2:3] = scale(trainingSet[, 2:3])
# testSet[, 2:3] = scale(testSet[, 2:3])
 
