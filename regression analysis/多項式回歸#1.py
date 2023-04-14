import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

dataset = pd.read_csv('C:/Users/User/Desktop/111-2 機器學習/Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
regeressor = LinearRegression()
regeressor.fit(x, y)


plot.scatter(x, y, color = 'red')
plot.plot(x, regeressor.predict(x), color = 'blue')
plot.title('Ture or Bluff (Linear Regression)')
plot.xlabel('Position Level')
plot.ylabel('Salary')
plot.show()


from sklearn.preprocessing import PolynomialFeatures
ploy_reg = PolynomialFeatures(degree = 2)
x_ploy = ploy_reg.fit_transform(x)
regeressor2 = LinearRegression()
regeressor2.fit(x_ploy, y)


plot.scatter(x, y, color = 'red')
plot.plot(x, regeressor2.predict(ploy_reg.fit_transform(x)), color = 'blue')
plot.title('Ture or Bluff (Polynomial Features)')
plot.xlabel('Position Level')
plot.ylabel('Salary')
plot.show()