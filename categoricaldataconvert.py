import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

dataset=pd.read_csv("data.csv")
print(dataset)
#using head and tail
print("top 5 rows")
print(dataset.head())
print("\nbottom 5 rows\n")
print(dataset.tail())
print("\ndatatype of cols\n")
print(dataset.dtypes)

print(dataset['Colour'])
print(dataset['Colour'].unique())

print("\nunique values in the column Country\n")
print(dataset['Country'].unique())

onehotencoderobj=OneHotEncoder()
print("object :",onehotencoderobj)

#converting into categorical values of columns colors and Country
converted_data=onehotencoderobj.fit_transform(dataset[['Colour','Country']]).toarray()
print(converted_data)
#seeing categories
print("\n")
print(onehotencoderobj.categories_)