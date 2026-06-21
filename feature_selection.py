import numpy as np
import pandas as pd
from preprocessing import data
from preprocessing import C_N
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
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
    'A': 0,
    'AT': 1,
    'T': 2
}

df['Size_Encoded'] = df['Size'].map(size_mapping)
# print(df["Size_Encoded"])
df['lesion']=C_N
df['pigment_network']=data["Pigment Network\n(AT/T)"]
df['streaks']=data["Streaks\n(A/P)"]
df['regression_area']=data["Regression Areas\n(A/P)"]
df['blue_white_veil']=data["Blue-Whitish Veil\n(A/P)"]
df=df.drop(columns=['Size'])

X_train,X_test,Y_train,Y_test=train_test_split(df[['Size_Encoded','pigment_network','streaks','regression_area','blue_white_veil']],df['lesion'],test_size=0.3,random_state=100)
# print(X_train.head(),X_test.head(),Y_train.head(),Y_test.head())

# print(X_train.isnull().sum())
# print(X_train)
f_p_values=chi2(X_train,Y_train)
# print(f_p_values)

p_values=pd.Series(f_p_values[1])
p_values.index=X_train.columns
# print(p_values)

print(p_values.sort_index(ascending=False))