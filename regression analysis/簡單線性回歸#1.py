import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

dataset = pd.read_csv('C:/Users/User/Desktop/111-2 機器學習/Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regeressor = LinearRegression()
regeressor.fit(x_train, y_train)

y_pred = regeressor.predict(x_test)


plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, regeressor.predict(x_train), color = 'blue')
plot.title('Salary VS Experience (training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

plot.scatter(x_test, y_test, color = 'red')
plot.plot(x_train, regeressor.predict(x_train), color = 'blue')
plot.title('Salary VS Experience (test set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()