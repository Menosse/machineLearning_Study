# SVR

# Importing dataset
SVRdataSet = read.csv("Position_Salaries.csv")
SVRdataSet = SVRdataSet[2:3]

# Installing SVR package
install.packages('1071', dep = TRUE, type = "source")

# Importing the library
library(e1071)

# Fitting SVR to the dataset
regressor = svm(formula = Salary ~ .,
                data = SVRdataSet,
                type = "eps-regression")

# Predicting a new result
yPredict = predict(regressor, data.frame(level = 6.5))

# Visualizing SRV results
library(ggplot2)
ggplot()+
  geom_point(aes(x = SVRdataSet$Level, SVRdataSet$Salary),
             colour = 'red') +
  geom_line(aes(x = SVRdataSet$Level, y = predict(regressor, newdata = SVRdataSet)),
            colour = 'blue') +
  ggtitle('Truth or bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')