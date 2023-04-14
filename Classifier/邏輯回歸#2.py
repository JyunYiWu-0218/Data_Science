# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('C:/Users/User/Desktop/111-2 機器學習/Iris.csv')
x = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, [5]].values
dataset.describe()

dataset.head()
dataset.info()
dataset.isna().sum()

# 進行分類型欄位編碼
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_y = LabelEncoder()
y[:, 0] = labelEncoder_y.fit_transform(y[:, 0])

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

# 創建一個 Neighbors Classifier 實例並擬合數據。
logreg.fit(x, y.astype('int'))

# 繪製決策邊界。為每種品種分配一種顏色
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
h = .02  # step size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# 將結果放入彩色圖中
Z = Z.reshape(xx.shape)
plot.figure(1, figsize=(4, 3))
plot.pcolormesh(xx, yy, Z, cmap=plot.cm.Paired)

# 繪製訓練點
plot.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plot.cm.Paired)
plot.xlabel('Sepal length')
plot.ylabel('Sepal width')

plot.xlim(xx.min(), xx.max())
plot.ylim(yy.min(), yy.max())
plot.xticks(())
plot.yticks(())

plot.show()
