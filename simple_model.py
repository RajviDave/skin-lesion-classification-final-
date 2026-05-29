from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import os

data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)

print(data.columns)
df=pd.DataFrame(data['Pigment Network\n(AT/T)'])
# Dots=data['Dots/Globules\n(A/AT/T)']
# Streaks=data['Streaks\n(A/P)']
# regression_area=data['Regression Areas\n(A/P)']
# veil=data['Blue-Whitish Veil\n(A/P)']
# df['Dots']=Dots
# df['Streaks']=Streaks
# df['Regression']=regression_area
# df['Veil']=veil

encoder=OrdinalEncoder(categories=[['T','AT']])
df['Pigmentation_Encoded']=encoder.fit_transform(df[['Pigment Network\n(AT/T)']])
print(df)
