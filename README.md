# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array. 
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.obtain the graph.


## Program:
```

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VELAN D
RegisterNumber:  212222040176
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print("Placement data")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print("print data")
data1

x = data1.iloc[:,:-1]
print("Data-status")
x

y = data1["status"]
print("data-status")
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
![274937983-c637c601-a65c-490a-b150-e59fd7ead20f](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/0204e9ab-4016-4098-a095-faa49f93dfcb)
![274938195-0ca792ab-9331-49da-971f-942451d53a8a](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/85b1879f-3c31-4a07-b824-ae889ca82f24)
![274938151-051f3e58-99cf-458c-9644-0c5d77a48060-1](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/02795250-e1fe-43d1-824c-86ecea466674)
![274938293-ba9a0a0d-73d7-44cb-8057-c78f80987cc9](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/143174fb-2379-4d2a-be8b-640033b7f867)
![274938369-f58c4a69-7509-475b-97f1-e0561dbf99ec](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/4126366f-ab38-4ffe-a50c-1219fc4a654c)
![274938450-7134dfb3-7bc9-42c3-8b07-ad63f1bcd6a8](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/56f3ef51-a7a5-41d9-a252-a19d9543d893)
![274938555-31cdb53c-11bf-4150-8738-a72303ca0736](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/6ffc4f97-564d-4879-a5c5-bbd4cae8e20f)
![274938627-e91e0f4c-a1c3-4b0e-8c32-d29cccdff8bf](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/468e5997-f814-4b72-9b7b-ae280c2bdfae)
![274938674-92d51ca5-dde9-4935-8493-c883a155e802](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/882c2265-8eee-4c2e-a5b1-13a6868b0c45)
![274938717-9ce3f0df-2f98-4e70-ab51-09367c9a5aa4](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/8d7a1e4e-a575-4f6e-9651-22b64818ce51)
![274938762-dfc725d5-f0d5-4cd5-9946-98bb5af4f231](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/10f973b0-fdde-4f34-859c-8ba13e71efb5)
![282254994-b8de09dc-da5b-4818-9f50-f07f92c31391](https://github.com/VELANDHANANJAYAN/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405038/90588153-f436-4027-b04b-980fbf9b1c67)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
