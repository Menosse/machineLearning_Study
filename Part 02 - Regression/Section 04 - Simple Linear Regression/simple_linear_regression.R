# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# Feature Scaling is not necessary for SIMPLE LINEAR regression
# trainingSet = scale(trainingSet)
# testSet = scale(testSet)

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = trainingSet)

# Predicting the Test set results
yPred = predict(regressor, newdata = testSet)

# Visualising the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = trainingSet$YearsExperience, y = trainingSet$Salary),
             colour = 'red') +
  geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualising the Test set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary),
             colour = 'red') +
  geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')