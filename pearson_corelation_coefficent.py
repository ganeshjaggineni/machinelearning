#importing libraries
import numpy as np
import matplotlib.pyplot as plt

print("correlation coefficent for simiarity")
arr1=np.array([12,14,19,24])
arr2=np.array([22,24,25,27])
similarcor=np.corrcoef(arr1,arr2)
print(similarcor)
plt.scatter(arr1,arr2)
plt.show()
print("correlation coefficent for disimilarity")
arr3=np.arange(3,8)
print("array arr3 ")
print(arr3)
arr4=np.array([55,25,15,14,9])
dissimilarcor=np.corrcoef(arr3,arr4)
print(dissimilarcor)
plt.scatter(arr3,arr4)
plt.show()