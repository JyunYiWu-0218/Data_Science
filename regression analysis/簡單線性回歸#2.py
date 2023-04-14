import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv("C:/Users/User/Desktop/111-2 機器學習/Customers.csv")
x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 4].values

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

dataset.head()
dataset.info()
dataset.isna().sum()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=42)


from sklearn.linear_model import LinearRegression
regeressor = LinearRegression()
regeressor.fit(x_train, y_train)

y_pred = regeressor.predict(x_test)

plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, regeressor.predict(x_train), color = 'blue')
plot.title('age VS Spending Score  (training set)')
plot.xlabel('age')
plot.ylabel('Spending Score')
plot.show()

plot.scatter(x_test, y_test, color = 'red')
plot.plot(x_train, regeressor.predict(x_train), color = 'blue')
plot.title('age VS Spending Score (test set)')
plot.xlabel('age')
plot.ylabel('Spending Score')
plot.show()


