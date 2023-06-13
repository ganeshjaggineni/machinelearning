import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dataset=pd.read_csv("Mall_Customers.csv")
print(dataset.head())
print(dataset.tail())
print(dataset.isna().sum())
x=dataset.iloc[:,3:].values
# print(x)
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
wcss
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("no.of clusters")
plt.ylabel("wcss")
plt.show()

kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label="cluster1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label="cluster2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label="cluster3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='violet',label="cluster4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='orange',label="cluster5")
plt.show()