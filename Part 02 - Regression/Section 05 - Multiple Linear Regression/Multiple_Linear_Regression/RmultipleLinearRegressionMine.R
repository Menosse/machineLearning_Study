# Multiple linear regression

# DATA PREPROCESSING

# IMPORT DATA SET
dataset = read.csv('50_Startups.csv')

# Encode categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

# Splitting the dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)
 
# # To represent the formula faster use the expression below
regressor = lm(formula = Profit ~ .,
               data = trainingSet)

# Predicting the Test set results
yPred = predict(regressor,
                newdata = testSet)

# Build an optimal model using Backward Elimination algorithym
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
#                data = dataset)
# summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)

# # Visualizing the Training set results
# install.packages("ggplot2")
# library(ggplot2)
# 
# ggplot() +
#   geom_point(aes(x = trainingSet$Profit, y = trainingSet$R.D.Spend),
#              colour = 'red') +
#   geom_line(aes(x = trainingSet$Profit, y = predict(regressor, newdata = trainingSet)),
#             colour = 'blue') +
#   ggtitle('Salary vs Experience(Training Set)') +
#   xlab('Investment') +
#   ylab('Profit')
# 
# 
# # Visualizing the Test set results
# # install.packages("ggplot2")
# # library(ggplot2)
# 
# ggplot() +
#   geom_point(aes(x = testSet$YearsExperience, y = testSet$Salary),
#              color = 'gold') +
#   geom_line(aes(x = trainingSet$YearsExperience, y = predict(regressor, newdata = trainingSet)),
#             color = 'black',
#             size = 1,
#             alpha = 1) +
#   ggtitle('Salary vs Experience(Test Set)') +
#   xlab('Years of Experience') +
#   ylab('Salary')