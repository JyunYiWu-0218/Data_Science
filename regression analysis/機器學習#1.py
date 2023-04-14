import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


imputer = SimpleImputer(missing = np.nan, strategy = 'mean', fill_value = None)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = transform(x[:, 1:3])


labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')

X = ct.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
