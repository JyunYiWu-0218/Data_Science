## 目錄   
-   目錄
-   [Data_Science 學習歷程(個人)](#learn)
-   [(Recommended) Basic Knowledge](#Knowledge)
-   [Introduction to Applied Technology](#Introduction)
    -   Machine Learning
    -   Deep Learing
-   [Foreword](#Foreword)
-   [Machine Learning](#Machine)
    -   Data Preprocessing
    -   Supervised Learning
        -   [x] Regression Analysis
        -   [x] Classification
    -   非監督式學習 (Unsupervised Learning)
        -   [x] Principal components analysis(PCA)
        -   [x] Cluster analysis
        -   [x] Generative Adversarial Network(GAN)
    -   Semi-supervised Learning
        -   [ ] Semi-supervised Generative Model
        -   [ ] Self-training
        -   [ ] Entropy-based Regularization
        -   [x] Cluster and then label
        -   [ ] Graph-based Approach
        -   [ ] Better Representation
    -   Reinforcement Learning
-   [Deep Learing](#Deep)
    -   [ ] Deep Neural Networks
    -   [ ] Recurrent Neural Network
    -   [ ] Deep Belief Networks
    -   [x] Convolutional Neural Networks

## Data_Science 學習歷程(個人)  
記錄學習 Data_Science 的過程，並分享常看到，常聽到，常用到的技術及背後觀念(以個人觀點出發，有不對的請給予批評及指教)  主要以 Python 為主    
<a name="learn"/>



## (Recommended) Basic Knowledge
To engage in AI-related research, of course, you must have a certain understanding of basic mathematical knowledge! (You can also be a package user or a coder)  
**The following are the required mathematical knowledge:**    
1.微積分(calculus)  
2.偏導函數、重積分(partial derivative function, reintegration)  
3.線性代數(Linear Algebra)    
4.機率與統計(Probability and Statistics)  
<a name="Knowledge"/>


## Introduction to Applied Technology
An introduction to machine learning and deep learning.  

### Machine Learning
機器學習是人工智慧的一種子方法，利用完整的一套流程，將資料整理、分析、視覺化，讓原本無聊的資料煥然一新的呈現在你我的眼前!      
![calsses](https://user-images.githubusercontent.com/128043244/229371106-baaa4e65-d9ff-4d05-8fd7-7947c825ca62.png)
![Regression](https://user-images.githubusercontent.com/128043244/229371194-2bee9c06-b909-42a4-a4e7-e8b55800d2e8.png)


### Deep Learing
利用神經網路(Neural Network)讓電腦能從事人類的行為或是人類的職業!  
![Kears](https://user-images.githubusercontent.com/128043244/229407974-adf2c83c-afa9-446b-9eeb-7603c2632ea7.png)
![LAS_and_CTC_NLP](https://user-images.githubusercontent.com/128043244/229408001-8bd5c59f-fe6d-4a73-9e1a-70242ec218c7.png)
![neural_network](https://user-images.githubusercontent.com/128043244/229408206-1cd5ffec-103f-4968-b5f2-6977ca45ded0.png)

<a name="Introduction"/>


## Foreword
機器學習 (Machine Learning) 與深度學習 (Deep Learing) 都是現在很熱門的技術，各自有一套標準的執行流程，會藉由接下來的主題 一 一 詮釋!  
<a name="Foreword"/>


## Machine Learning
In-Depth Introduction to Machine Learning (Theory).  

### Data Preprocessing (資料預處理)
資料不可能都是馬上就可以進行分析(analyze)及視覺化(visualize)的，現實生活中的資料遠遠比理想中的資料還要難處理，會遇到很多無法預期的事情(收集資料時發現資料過於離散、資料遺失、類別不大相同、型別(Type)不同等)，因此資料預處理就是一件極其重要的過程！  
<img src='https://www.ipt.fraunhofer.de/en/offer/digitization/big-data/data-quality/jcr:content/contentPar/sectioncomponent/sectionParsys/textwithasset/imageComponent/image.img.4col.large.png/1589185347498/datenqualitaet-datenvorverarbeitung-bild1.png'>
#### Missing value handling (缺失值處理)
**一般來說缺失值處理會用以下常見的方法進行處理:**  
***1. 直接剔除(Deletion)帶有缺失值(Missing value)的行(column)或列(raw):***    
此方法不建議頻繁使用，當資料之間具有一定相關性($-1 \leq \rho \leq 1$)時，貿然刪除資料會導致分析結果不佳。      
相關係數(Pearson’s correlation coefficient, $\rho$):    

<img src='https://latex.codecogs.com/png.image?%5Cinline%20%5Clarge%20%5Cdpi%7B200%7D%5Cbg%7Bwhite%7D%5Crho%20=%20%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft%20(%20x_%7Bi%7D-%5Cmu%20_%7Bx%7D%20%5Cright%20)%5Cleft%20(%20y_%7Bi%7D-%5Cmu%20_%7By%7D%20%5Cright%20)%7D%7B%5Csqrt%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft%20(%20x_%7Bi%7D-%5Cmu%20_%7Bx%7D%20%5Cright%20)%5E%7B2%7D%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft%20(%20y_%7Bi%7D-%5Cmu%20_%7By%7D%20%5Cright%20)%5E%7B2%7D%7D%7D'>

***2. 填補固定值(Dummy substitution):***  
replace missing values with fill_value. Can be used with strings or numeric data.  
If None, fill_value will be 0 when imputing numerical data and “missing_value” for strings or object data types.  
```python=
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'constant', fill_value = None)
```


***3. 填補平均值(Mean substitution):***  
replace missing values using the mean along each column. Can only be used with numeric data.  
```python=
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean', fill_value = None)
```  


***4. 填補高頻資料(Frequent substitution):***  
replace missing using the most frequent value along each column. Can be used with strings or numeric data.  
If there is more than one such value, only the smallest is returned.   
```python=
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent', fill_value = None)
```


***5. 填補中位數(Median substitution):***  
replace missing values using the median along each column. Can only be used with numeric data.  
```python=
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'median', fill_value = None)
```


#### Outlier handling (異常值處理)
異常值：偏離樣本整體數據的值    

##### 判斷異常值 (Judge outliers)
***1. 常識判斷：根據資料的特性去做判斷***  
  
  
***2. 基本統計方法：對資料進行描述性統計***  
  
<img src='https://image-static.segmentfault.com/306/690/3066903361-6011059e8a47e'>

$\mu \pm 3\sigma$,data need to be processed for outliers！  

```Python=
import numpy as np
def outliers_z_score(data,times):
    data_mean = np.mean(data)
    data_stdev = np.std(data)
    z_scores = [(i - data_mean) / data_stdev for i in x]
    return np.where(np.abs(z_scores) > times)
```

***3. 盒鬚圖判別法***  
  
<img src='https://www.finereport.com/tw/wp-content/uploads/2022/11/2022111401C.png'>


```python=
import seaborn as sns
import matplotlib.pyplot as plt
features=["A","B","C","D","E"]
fig, ax = plt.subplots()
fig.subplots_adjust(hspace=1, wspace=0.6)
location=1
for i in features:
    plt.subplot(2, 3, location)    
    sns.boxplot(data=train_df,x=i) 
    location+=1
```

#### Data Transformation (One-hot_Encoding and feature_scaling)

***One-hot_Encoding***  
<img src='https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ZFCX83XaMNzOAXRxAcvMJw.png'>
```python = 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[col, raw] = labelEncoder_x.fit_transform(x[col, raw])
ct = ColumnTransformer([("sex", OneHotEncoder(), [raw])] , remainder='passthrough')
x = ct.fit_transform(x)
```

***feature_scaling***    
<img src='https://python-data-science.readthedocs.io/en/latest/_images/scaling.png'>
1.Standard Scaler     
$$Z = \frac{\bar{X} - \mu }{\sigma / \sqrt{n}}$$  

```python=
import pandas pd
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# note that the test set using the fitted scaler in train dataset to transform in the test set
X_test_scaled = scaler.transform(X_test)
```

2.Min Max Scale  
$$X_{i} =\frac{X_{i} - min(X)}{max(X) - min(X)}$$

```python=
import pandas pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
```

3.Robust Scaling   
$$\bar{x} = \frac{X - median(X)}{75th quantlie(X) - 25th quantlie(X)}$$

```python=
robustscaler = RobustScaler() # create an object
X_train_scaled = robustscaler.fit_transform(X_train)
X_test_scaled = robustscaler.transform(X_test)
```

4.Normalizer(Scaling to unit length)  
Scales each data point such that the feature vector has a Euclidean length of 1.  
將變數值除以變數的歐幾里得距離(Euclidean distance)或曼哈頓距離(Manhattan distance)。  
$$\bar{x} = \frac{X}{\left \| X \right \|}$$  

**Euclidean distance**    
$$L2(X) = \sqrt{{x_{1}}^{2} + {x_{2}}^{2} + {x_{3}}^{2} + ... + {x_{n}}^{2}}$$

**Manhattan distance**  
$$L1(X) = \left | X_{1} \right | + \left | X_{2} \right | + ... + \left | X_{n} \right |$$


### Supervised Learning


### 非監督式學習 (Unsupervised Learning)


### Semi-supervised Learning


### Reinforcement Learning

<a name="Machine"/>


## Deep Learing
In-Depth Introduction to Deep Learning and Neural Networks (Theory).  

### Deep Neural Networks


### Recurrent Neural Network


### Deep Belief Networks


### Convolutional Neural Networks
<a name="Deep"/>
