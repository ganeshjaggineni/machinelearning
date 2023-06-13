import numpy as np
import pandas as pd
from apyori import apriori
data=pd.read_csv("mark.csv")
data.head()
#removing header
data=pd.read_csv("mark.csv",header=None)
print(data.head())
print(data.shape)
data.dropna()

records=[]
for i in range(0,7501):
  records.append([str(data.values[i,j]) for j in range(0,20)])
print(records)

association_rules=apriori(records,min_support=0.0045,min_confidence=0.2,min_lift=3,min_length=2)
association_result=list(association_rules)
# print(association_result)

print(association_result[0])

for item in association_result:
  pair=item[0]
  items=[x for x in pair]
  # print("items rule ",items)
  print('Rule :'+items[0]+" --> "+items[1])
  print("support = ",item[1])
  print("confidence =",item[2][0][2])
  print("+++++++++++++++++++++++++++++++++++++")