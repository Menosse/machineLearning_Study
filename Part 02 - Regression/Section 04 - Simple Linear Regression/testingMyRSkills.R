# Simple Linear Regression
# Dependent variable = Salary
# Independent variable = Years of experience

# Importing the dataset
newDataSet = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
set.seed(123)
split = sample.split(newDataSet, SplitRatio = 2/3)
trainSet = subset(newDataSet, split == TRUE)
testSet = subset(newDataSet, split == FALSE)

# Fitting Simple Linear Regression to the Training set
linReg = lm(formula = Salary ~ YearsExperience,
            data = trainSet)

# Predicting the Test set results
yPred = predict(linReg, newdata = testSet)

#Importing plot Library
library(ggplot2)
# Visualising the Training set results
ggplot()+
  ggtitle("Salary x Experience (Training Set)") +
  xlab('Experience') +
  ylab('Salary') +
  geom_point(aes(x = trainSet$YearsExperience, y = trainSet$Salary),
             colour = 'red') +
  geom_line(aes(x = trainSet$YearsExperience, y = predict(linReg, newdata = trainSet)),
            colour = 'blue')
  
  
  
# Visualising the Test set results
  ggplot()+
  ggtitle("Salary x Experience (Training Set)") +
  xlab('Experience') +
  ylab('Salary') +
    geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary),
               colour = 'blue') +
    geom_line(aes(x = testSet$YearsExperience, y = predict(linReg, newdata = testSet)),
              colour = 'red')
