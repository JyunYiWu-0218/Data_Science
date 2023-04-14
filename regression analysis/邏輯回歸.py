import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

dataset = pd.read_csv('C:/Users/User/Desktop/111-2 機器學習/Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

dataset.head()
dataset.info()
dataset.isna().sum()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state= 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc_X = StandardScaler() # create an object
x_train = sc_X.fit_transform(x_train) 
x_test = sc_X.transform(x_test)


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)