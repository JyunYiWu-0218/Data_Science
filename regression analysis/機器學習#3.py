import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Customers.csv')
x = dataset.iloc[:, [1, 2, 3, 5, 6, 7]].values
y = dataset.iloc[:, 4].values

dataset.head()
dataset.info()
dataset.isna().sum()

from sklearn.impute import SimpleImputer

#補齊缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None)
imputer = imputer.fit(x[:,[5]])
x[:,[5]] = imputer.transform(x[:,[5]])

Y = np.reshape(y, (-1,1))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 編成數字碼
labelencoder_x = LabelEncoder()
x[:,1] = labelencoder_x.fit_transform(x[:,1])
ct=ColumnTransformer([("gender", OneHotEncoder(), [1])]  ,  remainder='passthrough')
x[:,5] = labelencoder_x.fit_transform(x[:,5])
ct=ColumnTransformer([("profession", OneHotEncoder(), [5])]  ,  remainder='passthrough')
X = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

