import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Customer_Behaviour.csv')
x = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

dataset.head()
dataset.info()
dataset.isna().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

imputer2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None)
imputer2 = imputer2.fit(x[:,0:1])
x[:,0:1] = imputer2.transform(x[:,0:1])


Y = np.reshape(y, (-1,1))
imputer2 = imputer2.fit(Y[:,:])
Y[:,:] = imputer2.transform(Y[:,:])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:,[1, 5]] = labelencoder_x.fit_transform(x[:,[1, 5]])

ct=ColumnTransformer([("gender", OneHotEncoder(), [0])]  ,  remainder='passthrough')

X = ct.fit_transform(x)

# =============================================================================
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# =============================================================================

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)