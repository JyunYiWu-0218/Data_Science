import numpy as np
import matplotlib.pyplot as plot
import pandas as pd


dataset = pd.read_csv("C:/Users/User/Desktop/111-2 機器學習/insurance.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

dataset.head()
dataset.info()
dataset.isna().sum()

y = y.reshape(-1, 1)



#補缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan , strategy = 'most_frequent', fill_value = None)
imputer = imputer.fit(x[:, [0, 1, 2, 3, 4]])
x[:, [0, 1, 2, 3, 4]] = imputer.transform(x[:, [0, 1, 2, 3, 4]])



imputer_y = SimpleImputer(missing_values= np.nan , strategy = 'mean', fill_value = None)
imputer_y = imputer_y.fit(y[:,])
y[:,] = imputer_y.transform(y[:,])


#處理分類型欄位
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[:, 1] = labelEncoder_x.fit_transform(x[:, 1])
ct = ColumnTransformer([("sex", OneHotEncoder(), [1])] , remainder='passthrough')
x = ct.fit_transform(x)

x[:, 5] = labelEncoder_x.fit_transform(x[:, 5])
ct = ColumnTransformer([("smoker", OneHotEncoder(), [5])] , remainder='passthrough')
x = ct.fit_transform(x)

x[:, 7] = labelEncoder_x.fit_transform(x[:, 7])
ct = ColumnTransformer([("region", OneHotEncoder(), [7])] , remainder='passthrough')
x = ct.fit_transform(x)

x = x[:, :-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state= 0)

from sklearn.linear_model import LinearRegression #不須做變量縮放原因:sklearn 線性回歸函式已經有做處理
regeressor = LinearRegression()
regeressor.fit(x_train, y_train)

y_pred = regeressor.predict(x_test)

#反向淘汰
import statsmodels.api as stat
x_train = np.append(arr = np.ones((1070, 1)).astype(int), values = x_train, axis = 1)
x_opt = x_train[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 1, 2, 3, 4, 6, 8, 9]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 1, 2, 3, 6, 8, 9]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()

x_opt = x_train[:, [0, 1, 2, 3, 8, 9]]
x_opt = np.array(x_opt, dtype = float)
regeressor_OLS = stat.OLS(endog = y_train, exog = x_opt).fit()
regeressor_OLS.summary()