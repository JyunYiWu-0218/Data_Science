import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv("C:/Users/User/Desktop/111-2 機器學習/50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[:, 3] = labelEncoder_x.fit_transform(x[:, 3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])] , remainder='passthrough')
x = ct.fit_transform(x)

x = x[:, :-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state= 0)

from sklearn.linear_model import LinearRegression #不須做變量縮放原因:sklearn 線性回歸函式已經有做處理
regeressor = LinearRegression()
regeressor.fit(x_train, y_train)

y_pred = regeressor.predict(x_test)

import statsmodels.api as stat
x_train = np.append(arr = np.ones((40, 1)).astype(int), values = x_train, axis = 1)
x_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 1, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 4]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()
