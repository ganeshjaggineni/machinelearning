import numpy as np
import pandas as pd

from apyori import apriori

data=pd.read_csv("market.csv")
print("\ndataset\n")
print(data)

basket=[]
for i in range(1,7500):
  basket.append([str(data.values[i,j]) for j in range(0,20)])

#apriori rules

association_rules=apriori(data,min_support=0.0045,min_confidence=0.2,min_lift=3,min_lenght=2)
association_result=list(association_rules)

print("association_result")
print(association_result)

for i in range(0,len(associatin_results)):
  print(association_result[i][0])