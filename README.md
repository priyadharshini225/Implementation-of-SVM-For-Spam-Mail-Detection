# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather a labeled dataset containing both spam and non-spam emails, with labels typically as 0 (non-spam) and 1 (spam)
2. Clean the email text by removing punctuation, converting text to lowercase, and removing stop words to reduce noise.
3. Split the dataset into training and testing sets, typically using an 80-20 or 70-30 ratio.
4. Train the SVM on the training data, allowing it to learn patterns associated with spam and non-spam emails.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYADHARSHINI S 
RegisterNumber: 212223240129
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
df=pd.read_csv("spam.csv",encoding='Windows-1252')

df.head()

df.info()

df.isnull().sum()

x=df['v2'].values
y=df['v1'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#CountVectorizer is convert text into numerical data

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
## Head():
![Screenshot 2024-11-14 223912](https://github.com/user-attachments/assets/f3879ef0-822d-4788-9714-abf408781585)

## Info():
![Screenshot 2024-11-14 223918](https://github.com/user-attachments/assets/8b6ecc8e-5ad4-4b18-883f-77cfbf53fbac)

## Y-Predict:
![Screenshot 2024-11-14 223852](https://github.com/user-attachments/assets/c9aaf67b-529d-4ea4-9ce9-9910acefbb83)


## Accuracy:
![Screenshot 2024-11-14 223857](https://github.com/user-attachments/assets/ed90aa85-544b-41ae-84fa-be9234571e7f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
