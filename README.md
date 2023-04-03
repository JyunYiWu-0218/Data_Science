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
![Data_Preprocessing](https://user-images.githubusercontent.com/128043244/229455675-511abcb6-c8fb-4371-b89b-3e308ae18832.png)
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
