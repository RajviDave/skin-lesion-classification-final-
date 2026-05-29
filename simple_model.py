from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import os

data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)

print(data.columns)
