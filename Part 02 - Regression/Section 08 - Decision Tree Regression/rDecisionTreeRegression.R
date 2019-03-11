# Autho - Fernando Takada
# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
# Fitting the Regression Model to the dataset
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression results (for higher resolution and smoother curve)
library(ggplot2)
xGrid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = xGrid, y = predict(regressor, newdata = data.frame(Level = xGrid))),
            colour = "blue")+
  ggtitle('Truth or Bluff (Decision Tree Regression Model)')+
  xlab('Position level')+
  ylab('Salary')