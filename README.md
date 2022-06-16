# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown value
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Subanu.K
RegisterNumber:  212219040152
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
**Original data(first five columns):**

![r1](https://user-images.githubusercontent.com/87663343/174066212-ac0d404f-37ee-44fd-b8b4-020cc674a7a8.png)

**Data after dropping unwanted columns(first five):**

![r2](https://user-images.githubusercontent.com/87663343/174066422-fc4823a4-f9fe-44ce-a8b1-e77c45fc739f.png)

**checking the presence of null values:**

![r3](https://user-images.githubusercontent.com/87663343/174066544-81ec1fe5-f237-432d-864b-ac754d729c12.png)

**checking the presence of duplicated values:**

![r4](https://user-images.githubusercontent.com/87663343/174066684-2af2ffb8-f2ad-4846-94e6-74542c36b6c8.png)

**Data after encoding:**

![r5](https://user-images.githubusercontent.com/87663343/174066903-5f205e5d-7e24-458f-a001-2f907d0beb91.png)

**X data:**

![r6](https://user-images.githubusercontent.com/87663343/174067015-1d326834-d9fc-44fe-9649-acd45f4c434a.png)

**Y data:**

![r7](https://user-images.githubusercontent.com/87663343/174067121-3f2c146d-3dd5-49ea-829e-f447e698e514.png)

**predicted values:**

![image](https://user-images.githubusercontent.com/87663343/174067238-2c35da25-57e4-4a0a-93eb-e41a264a93ec.png)

**Accuracy score:**

![image](https://user-images.githubusercontent.com/87663343/174067333-42ff9a06-d4c7-4853-8f3c-ac7c536964ce.png)

**Confusion matrix:**

![image](https://user-images.githubusercontent.com/87663343/174067396-d715abca-0e66-4c24-8ba5-7ca030cd50c7.png)

**classification report:**

![image](https://user-images.githubusercontent.com/87663343/174067499-b2ba68a7-2419-469c-ba8b-1d59b209c0b7.png)

**Predicting Output from regression model:**

![image](https://user-images.githubusercontent.com/87663343/174067630-a4be1cc6-abbc-44b4-82d4-ac0c576a5be2.png)















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
