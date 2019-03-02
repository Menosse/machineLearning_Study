# Simple linear regression

# DATA PREPROCESSING

# IMPORT DATA SET
dataset = read.csv('Salary_Data.csv')

# #Splitting the dataset into training set and test set

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# # Feature Scaling - !!Simple linear regression does not need feature scaling!!
# trainingSet[, 2:3] = scale(trainingSet[, 2:3])
# testSet[, 2:3] = scale(testSet[, 2:3])
 
#Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = trainingSet)

# Predicting the Test set results
yPred = predict(regressor,
                newdata = testSet)

# Visualizing the Training set results
install.packages("ggplot2")
library(ggplot2)

ggplot() +
  geom_point(aes(x = trainingSet$YearsExperience, y = trainingSet$Salary),
             colour = 'red') +
  geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)),
            colour = 'blue') +
  ggtitle('Salary vs Experience(Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')


# Visualizing the Test set results
# install.packages("ggplot2")
# library(ggplot2)

ggplot() +
  geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary),
             color = 'gold') +
  geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)),
            color = 'black',
            size = 1,
            alpha = 1) +
  ggtitle('Salary vs Experience(Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')