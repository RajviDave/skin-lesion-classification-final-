import numpy as np
import pandas as pd
from preprocessing import data
from preprocessing import C_N
from sklearn.preprocessing import OrdinalEncoder
# print(C_N)
# print(data.columns)

data["Pigment Network\n(AT/T)"]=np.where(data["Pigment Network\n(AT/T)"]=="AT",0,1)
# print(data["Pigment Network\n(AT/T)"])

data["Streaks\n(A/P)"]=np.where(data["Streaks\n(A/P)"]=="A",0,1)
# print(data["Streaks\n(A/P)"])

data["Regression Areas\n(A/P)"]=np.where(data["Regression Areas\n(A/P)"]=="A",0,1)
# print(data["Regression Areas\n(A/P)"])

data["Blue-Whitish Veil\n(A/P)"]=np.where(data["Blue-Whitish Veil\n(A/P)"]=="A",0,1)
# print(data["Blue-Whitish Veil\n(A/P)"])

data_frame={"Size":data["Dots/Globules\n(A/AT/T)"]}
df = pd.DataFrame(data_frame)
size_mapping = {
    'A': -1,
    'AT': 0,
    'T': 1
}

df['Size_Encoded'] = df['Size'].map(size_mapping)
print(df["Size_Encoded"])