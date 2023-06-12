#cosine similarity
#importing libraries
import numpy as np
arr1=np.array([1,5,3,4,2,0,0,1])
arr2=np.array([0,0,1,3,5,6,0,2])
numerator=np.dot(arr1,arr2)
denominator=np.sqrt(sum(np.square(arr1)))*np.sqrt(sum(np.square(arr2)))
print("numerator ",numerator)
print("denominator ",denominator)
cosinesimilarity=numerator/denominator
print(cosinesimilarity)


#output
"""
numerator  27
denominator  64.80740698407861
0.4166190448976481
"""