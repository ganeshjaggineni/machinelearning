#naive bayes classification
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataset=pd.read_csv("play_tennis.csv")
#printing dataset
print(dataset)
dataset=dataset.drop(columns='day')
#removing unwanted columns
print("\n\ndataset after removing unwanted columns")
print(dataset)
#converting categorical data into numerical data
lab=LabelEncoder()
dataset['outlook']=lab.fit_transform(dataset['outlook'])
print(dataset)
dataset['temp']=lab.fit_transform(dataset['temp'])
print(dataset)
hum=dataset['humidity']
hum=lab.fit_transform(hum)
dataset['humidity']=hum
print(hum)
print(dataset)
#converting wind into numerical
win=dataset['wind']
win=lab.fit_transform(win)
print(win)
dataset['wind']=win
print(dataset)
#independent variables
x=dataset.iloc[:,:-1]
print("\nindependent variable\n\n")
print(x)
#dependent variable
y=dataset.iloc[:,-1:]
print("\ndependent variable\n\n")
print(y)
#testing and training data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=100)
print("\nx_train")
print(x_train)
print("\nx_train shape = ",x_train.shape)
print("\nx_test")
print(x_test)
print("\nx_test shape = ",x_test.shape)
print("\ny_train")
print(y_train)
print("\ny_train shape = ",y_train.shape)
print("\ny_test")
print(y_test)
print("\ny_test shape = ",y_test.shape)
#building model
gnbclassifier=GaussianNB()
print("gnbclassifer = ",gnbclassifier)
gnbclassifier.fit(x_train,y_train)
y_pred=gnbclassifier.predict(x_test)
print("\ny_original_test\n",y_test)
print("\ny_predicted_test\n",y_pred)
#finding accuracy
accuracy=metrics.accuracy_score(y_pred,y_test)*100
print("accuracy = ",accuracy)

val1=[[2,1,0,0]]
val2=[[0,0,1,0]]

def pred(val):
  y_pred_val=gnbclassifier.predict(val)
  print("\n\nplay tennis \t = ",y_pred_val)
pred(val1)
pred(val2)