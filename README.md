# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5.Print all the outputs.
6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: LATHIKA L J
RegisterNumber:  212223220050
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.tail()
data.info()
data.isnull().sum()
x=data['v2'].values
y=data['v1'].values
y.shape
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
y_train.shape
x_test.shapefrom sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)  
x_test = cv.transform(x_test)
x_train.shape
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
x_train.shapex_train.shape
```

## Output:
## head
![image](https://github.com/user-attachments/assets/d03a5251-c089-4134-8600-da9f4bfeb387)
## info
![image](https://github.com/user-attachments/assets/7d3c63fb-9cb2-4f1c-b5e5-e3654c5d52be)
## tail
![image](https://github.com/user-attachments/assets/fac01c0b-2895-4724-b663-c424822e1512)
## predicted value
![image](https://github.com/user-attachments/assets/7a25dac4-d201-4127-8b23-ef46456ca483)
## accuracy
![image](https://github.com/user-attachments/assets/64a669c7-3516-4ea1-b624-70b64440cf2b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
