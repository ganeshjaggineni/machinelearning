#linear regression
#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import *;
from sklearn.linear_model import LinearRegression;
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv("Salary_Data.csv")
print(dataset)
#independent variable
x=dataset.iloc[:,:-1]
print("\nindependent variable")
print(x)
#dependent variable
y=dataset.iloc[:,-1]
print("\ndependent variable")
print(y)
print(dataset.isna().sum())
#splitting dataset into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
print("x_train \n",x_train)
print("x_test \n",x_test)
print("y_train \n",y_train)
print("y_test\n",y_test)
#logic for linear regression
linearregressor=LinearRegression()
print("regression obj = ",linearregressor)
linearregressor.fit(x_train,y_train)
#predicting values
y_pred=linearregressor.predict(x_test)
print("\nprediction")
print("y_test")
print(y_test)
print("\ny_pred")
print(y_pred)

#graph for x_train dataset
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,linearregressor.predict(x_train),color="green")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
#graph for x_test dataset
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,linearregressor.predict(x_test),color="pink")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()