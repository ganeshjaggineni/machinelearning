import numpy as np
import pandas as pd
from sklearn import cluster
ratings=[['gani',5,6,1,2,4,5,6,7],['suresh',4,5,7,6,1,2,6,2],['appi',1,2,5,5,6,7,2,8]]
titles=['user','parasite','openheimer','dunkirk','intestellar','inception','ironman','thor','rrr']
movies=pd.DataFrame(ratings,columns=titles)
# print(movies)
from sklearn import cluster
data=movies.drop('user',axis=1)
print(data)
k_means=cluster.KMeans(n_clusters=2,max_iter=50,random_state=1)
k_means.fit(data)
labels=k_means.labels_
pd.DataFrame(labels,index=movies.user,columns=['Cluster Id'])
centroids=k_means.cluster_centers_
pd.DataFrame(centroids,columns=data.columns)
testdata=np.array([[4,2,1,2,3,4,5,6],[1,3,6,2,7,2,8,3],[1,2,6,3,7,3,7,8]])
labels=k_means.predict(testdata)
print(labels)
username=np.array([['sid'],['gani'],['chippa']]).reshape(-1,1)
print(username)
cols=movies.columns.tolist()
cols.append('cluster id')
newusers=pd.DataFrame(np.concatenate((username,testdata,labels),axis=1),columns=cols)
print(newusers)