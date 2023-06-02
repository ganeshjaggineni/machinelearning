import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree


dataset=pd.read_csv("Iris.csv")
print(dataset)

#preprocessing 
#droping column 1
print(dataset.isna().sum())
dataset=dataset.drop(columns='Id')
print("\ndataset after dropping column 1\n")
print(dataset)
print("\ndependent variable")
y=dataset['Species']
print(y)
print("\nIndependent variables\n")
x=dataset.drop(columns=['Species'],axis=1)
print(x)

#creating encoder object
model=LabelEncoder()
y=model.fit_transform(y)
y

#seperating into trainig and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
print("\ntraing  dataset")
print(x_train,y_train)
print("\ntesting dataset\n")
print(x_test,y_test)
print("\nshape of training dataset")
print("x train :",x_train.shape,"y train :",y_train.shape)

#decisiong tree classifer object
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)
print("\npredicting")
y_pred=clf.predict(x_test)
print("y_pred = ",y_pred)

print("Accuracy :",(y_pred,x_test)*100)

#visualizing model
tree.plot_tree(clf)
plt.show()