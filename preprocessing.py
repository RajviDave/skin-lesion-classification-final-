import pandas as pd
import os

images={"Images":os.listdir("DATASET/dermoscopic")}
df=pd.DataFrame(images)
data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)
data.dropna()

C_N=data['Common Nevus'].tolist()
A_N=data['Atypical Nevus'].tolist()
M=data['Melanoma'].tolist()
C_N=pd.Series(C_N)
A_N=pd.Series(A_N)
M=pd.Series(M)

C_N=C_N.dropna()
A_N=A_N.dropna()
M=M.dropna()

C_N=C_N.replace('X','Common Nevus')
A_N=A_N.replace('X','Atypical Nevus')
M=M.replace('X','Melanoma')

C_N=C_N.tolist()
A_N=A_N.tolist()
M=M.tolist()
print(type(M))

C_N.append(M)
print(C_N)