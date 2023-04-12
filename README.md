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
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'constant', fill_value = None)
```


***3. 填補平均值(Mean substitution):***  
replace missing values using the mean along each column. Can only be used with numeric data.  
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean', fill_value = None)
```  


***4. 填補高頻資料(Frequent substitution):***  
replace missing using the most frequent value along each column. Can be used with strings or numeric data.  
If there is more than one such value, only the smallest is returned.   
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent', fill_value = None)
```


***5. 填補中位數(Median substitution):***  
replace missing values using the median along each column. Can only be used with numeric data.  
```python
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

```Python
import numpy as np
def outliers_z_score(data,times):
    data_mean = np.mean(data)
    data_stdev = np.std(data)
    z_scores = [(i - data_mean) / data_stdev for i in x]
    return np.where(np.abs(z_scores) > times)
```

***3. 盒鬚圖判別法***  
  
<img src='https://www.finereport.com/tw/wp-content/uploads/2022/11/2022111401C.png'>


```python
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
```python 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[col, raw] = labelEncoder_x.fit_transform(x[col, raw])
ct = ColumnTransformer([("sex", OneHotEncoder(), [raw])] , remainder='passthrough')
x = ct.fit_transform(x)
```

***feature_scaling***  

<img src='https://mkang32.github.io/images/2020-12-27-feature-scaling/scaling.png'>

1.Standard Scaler     
$$Z = \frac{\bar{X} - \mu }{\sigma / \sqrt{n}}$$  

```python
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

```python
means = X_train.mean(axis = 0)
max_min = X_train.max(axis = 0) - X_train.min(axis = 0)

X_train_scaled = (X_train - means) / max_min
X_test_scaled = (X_test - means) / max_min
```

```python
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

```python
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


### 監督式學習(Supervised Learning)  
- Regression Analysis    
**1. Simple Linear Regression**  
使用時機：當只有單一變數時。  
方程式：
      $$\hat{y_{i}} = b_{0} + b_{1} * x$$ 

假設檢定：  
1.顯著性檢定(F test): 探討迴歸模型中的 $\beta$ 係數是否全部為0  

$$\begin{cases}
 & \text{} H_{0}: \beta_{1}=\beta_{2}=\beta_{n}=0 \\ 
 & \text{} H_{1}: \beta \neq 0(\beta_{1}or\beta_{2}or\beta_{n}) \\
\end{cases}$$  

2.邊際檢定(t test): 個別自變數之 $\beta$ 係數是否為0  

$$\begin{cases}
 & \text{} H_{0}: \beta_{n}=0 \\ 
 & \text{} H_{1}: \beta_{n} \neq 0 \\
\end{cases}$$  


利用 SSR 來選出最佳預測線(當 SSR 達到最小值即可找出)：           

$$SSR = \sum_{i=1}^{n}{\varepsilon_{i}}^{2} = \sum_{i=1}^{n}\left ( y_{i} - \hat{y_{i}} \right )^{2}$$

對 $b_{0}$ 及 $b_{1}$ 進行偏微分：  

$$\frac{\partial SSR}{\partial b_{0}} = \frac{\partial}{\partial b_{0}} \sum_{i=1}^{n}\left ( y_{i}-b_{0}-b_{1}x_{i} \right )^{2}$$

$$b_{0}=\bar{y}-b_{1}\bar{x}$$

$$\frac{\partial SSR}{\partial b_{1}} = \frac{\partial}{\partial b_{1}} \sum_{i=1}^{n}\left ( y_{i}-b_{0}-b_{1}x_{i} \right )^{2}$$

將 $b_{0}$ 代入 $b_{1}$ 進行運算：      

<img src='https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%5Cbg%7Bwhite%7Db_%7B1%7D=%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20y_%7Bi%7D-%5Cbar%7By%7D%20%5Cright%20)x_%7Bi%7D%7D%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20x_%7Bi%7D-%5Cbar%7Bx%7D%20%5Cright%20)x_%7Bi%7D%7D=%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20y_%7Bi%7Dx_%7Bi%7D-n%5Cbar%7By%7D%5Cbar%7Bx%7D%20%5Cright%20)%7D%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20x_%7Bi%7Dx_%7Bi%7D-n%5Cbar%7Bx%7D%5Cbar%7Bx%7D%20%5Cright%20)%7D=%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20y_%7Bi%7D-%5Cbar%7By%7D%20%5Cright%20)%5Cleft%20(%20x_%7Bi%7D-%5Cbar%7Bx%7D%20%5Cright%20)%7D%7B%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft(%20x_%7Bi%7D-%5Cbar%7Bx%7D%20%5Cright%20)%5E%7B2%7D'>

$b_{1}$ 為斜率： $b_{1} > 0$ , $\hat{y_{i}}$ 隨著 $x_{i}$ 上升而增加； $b_{1} < 0$ , $\hat{y_{i}}$ 隨著 $x_{i}$ 增加而減少    
$b_{0}$ 為截距： 當 $x_{i} = 0$ 時, $b_{0}$ 便相當於 $\hat{y_{i}}$ 的平均值； 當 $x_{i} \neq 0$ 時, 則 $b_{0}$ 無意義   

```python
def simple_linear_regression(raw_x, raw_y):
    n = np.size(raw_x)  #set the size of   
    x = np.array(raw_x)
    y = np.array(raw_y)
    x_mean = np.mean(x) #average
    y_mean = np.mean(y)

    num1 = np.sum(y*x) - n*y_mean*x_mean
    num2 = np.sum(x*x) - n*x_mean*x_mean
     
    b_1 = num1 / num2
    b_0 = y_mean - b_1 * x_mean
    
    return (b_0, b_1)
```
```python
# 使用 sklearn 實作
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=42) #Set test data set and training data set (8:2)


from sklearn.linear_model import LinearRegression
regeressor = LinearRegression()
regeressor.fit(x_train, y_train)

y_pred = regeressor.predict(x_test)

#畫圖
import matplotlib.pyplot as plot
plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, regeressor.predict(x_train), color = 'blue')
plot.title('age VS Spending Score  (training set)')
plot.xlabel('age')
plot.ylabel('Spending Score')
plot.show()
```

    誤差項 $\varepsilon _{0}$ 三大假設:      
    1. 常態性(Normality)：若母體為常態分配，則遵循(採用常態機率圖 normal probability plot 或 Shapiro-Wilk常態性檢定做檢查)     
    2. 獨立性(Independency)：誤差項需相互獨立(Durbin-Watson test來檢查)  
    3. 變異數同質性(Constant Variance)：變異數若不相等會導致自變數無法有效估計依變數  

    自變數 (Independent variable) 與 因變數 (Dependent variable):        
    1. 自變數 (Independent variable): 獨立的變數，會影響因變數及預測結果    
    2. 因變數 (Dependent variable)：依賴於自變數，通常設定為要預測的項目  



假設檢定 (Hypothesis Testing):  
將事件假設為 虛無假設( $H_{0}$ ) 與 對立假設( $H_{1}$ ),並確定假設(左尾、右尾、雙尾),利用 p值判斷是否成立 ( $p \leq \alpha(0.05)$ 成立假設)  
1. 虛無假設( $H_{0}$ ) : 希望能證明為錯誤  
2. 對立假設( $H_{1}$ ) : 透過假設檢定來證明對立假說為真，有充足證據拒絕虛無假說時，即可接受對立假說，而若無充足證據證明對立假說為真時，則「不拒絕」虛無假說
    
<img src="https://pic.pimg.tw/yourgene/1484708489-54953476.png">

Assuming 10  
雙尾檢定(Two-tailed test)：      

$$\begin{cases}
 & \text{} H_{0}: \mu = 10 \\
 & \text{} H_{1}: \mu \neq  10 \\
\end{cases}$$

左尾檢定(Left-tailed test)：        

$$\begin{cases}
 & \text{} H_{0}: \mu \geq  10 \\
 & \text{} H_{1}: \mu <  10 \\
\end{cases}$$

右尾檢定(Right-tailed test)：      

$$\begin{cases}
 & \text{} H_{0}: \mu \leq   10 \\
 & \text{} H_{1}: \mu >  10 \\
\end{cases}$$  

<img src="https://pic.pimg.tw/yourgene/1484708489-228202176_n.png">

       
顯著水準(significant level, $\alpha$ ):  
拒絕了「實際上成立的虛無假設」之機率，即犯下「Type 1 Error」的機率  
    
<!-- $$ P\left ( \bar{x}-z_{\alpha /2}*\frac{\sigma }{\sqrt{n}} \leq \mu \leq \bar{x}-z_{\alpha /2}*\frac{\sigma }{\sqrt{n}}\right ) = 0.95 $$ -->
<img src='https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%5Cbg%7Bwhite%7DP%5Cleft%20(%20%5Cbar%7Bx%7D-z_%7B%5Calpha%20/2%7D*%5Cfrac%7B%5Csigma%20%7D%7B%5Csqrt%7Bn%7D%7D%20%5Cleq%20%5Cmu%20%5Cleq%20%5Cbar%7Bx%7D-z_%7B%5Calpha%20/2%7D*%5Cfrac%7B%5Csigma%20%7D%7B%5Csqrt%7Bn%7D%7D%5Cright%20)%20=%200.95'>          
          
(公式為 95% 信賴區間的情況， $\alpha$ 通常情況為 0.05)  

***2. multivariate regression***  
使用時機：兩個以上的自變數(x)。(y 必須連續)    
方程式：  

$$y=a+b_{1}x_{1}+b_{2}x_{2}+.... +b_{n}x_{n}$$

模型:  

$$y=\beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+ ... +\beta_{n}x_{n}+\epsilon$$

- $\beta_{0}$ 為截距  
- $\beta_{1}$ ~ $\beta_{n}$ 為回歸係數(需估計)  
- $\epsilon$ 誤差項   

估計式:  

$$\hat{y}=\hat{\beta_{1}}x_{1}+\hat{\beta_{2}}x_{2}+.... +\hat{\beta_{n}}x_{n}$$



***3. Log-linear regression***   
***4. Log probability regression***  
***5. Nonlinear regression***   
***6. Partial regression***  
***7. (自我回歸)Ego regression***    
    - Autoregressive Moving Average Model
    - (差分)Differential Autoregressive Moving Average Model
    - Vector autoregressive models

- Classification   
***1. Logistic Regression***  
***2. Support Vector Machine(SVM)***  
***3. Naive Bayes classifier***  
***4. Decision Tree***  


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
