# Polinomial regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fittin linear regression to the Dataset
linReg = lm(formula = Salary ~.,
            data = dataset)
# Fitting Polynomial regression to the Dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
polyReg = lm(formula = Salary ~.,
             data = dataset)

# Install graphs library
install.packages('ggplot2')

# Visualize linear regression model
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(linReg, newdata = dataset)),
            colour = 'blue')+
  ggtittle('Truth or bluff (Linear Regression)')+
  xlab('Level') +
  ylab('Salary')

# Visualize Polynomial regression model
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(polyReg, newdata = dataset)),
            colour = 'blue') +
  ggtittle('Truth or bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predict a new result using linear regression
yPred = predict(linReg, data.frame(Level = 6.5))

# Creating new dataset
newDataSet = data.frame(Level = 6.5)
yPred = predict(linReg, newDataSet)

# Predict a new result using polynomial regression
yPred = predict(polyReg, data.frame(Level = 6.5,
                                    Level2 = 6.5^2,
                                    Level3 = 6.5^3,
                                    Level4 = 6.5^4))

# Creating new dataset
newDataSet = data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4)
yPred = predict(polyReg, newDataSet)