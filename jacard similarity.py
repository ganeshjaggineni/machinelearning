#jacard similarity
#importing libraries
import numpy as np
d1=np.array([10,12,13,15])
d2=np.array([12,10,11,15])
numerator=len((set(d1)).intersection(set(d2)))
print("numerator = ",numerator)
denominator=len(set(d1).union(set(d2)))
print("denominator = ",denominator)
jacardsimilarity=numerator/denominator
print("jacard similarity = ",jacardsimilarity)

#output
"""
numerator =  3
denominator =  5
jacard similarity =  0.6
"""