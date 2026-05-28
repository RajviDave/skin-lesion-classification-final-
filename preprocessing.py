import pandas as pd
import os

images={"Images":os.listdir("DATASET/dermoscopic")}
df=pd.DataFrame(images)
data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)
data.dropna()
print(data)
# C_N=data['Common Nevus'].tolist()
# A_N=data['Atypical Nevus'].tolist()
# M=data['Melanoma'].tolist()



