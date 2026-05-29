from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import os

data=pd.read_excel('PH2Dataset/PH2_dataset.xlsx',skiprows=12)

print(data.columns)
df=pd.DataFrame()

df['Pigment']=data['Pigment Network\n(AT/T)']
df['Dots']=data['Dots/Globules\n(A/AT/T)']
df['Streaks']=data['Streaks\n(A/P)']
df['Regression']=data['Regression Areas\n(A/P)']
df['Veil']=data['Blue-Whitish Veil\n(A/P)']

encoder=OrdinalEncoder(categories=[
    ['T', 'AT'],        # Pigment
    ['A', 'T', 'AT'],   # Dots
    ['A', 'P'],         # Streaks
    ['A', 'P'],         # Regression
    ['A', 'P']          # Veil
])

encoded=encoder.fit_transform(df)

encoded_df = pd.DataFrame(
    encoded,
    columns=[
        'Pigment_Encoded',
        'Dots_Encoded',
        'Streaks_Encoded',
        'Regression_Encoded',
        'Veil_Encoded'
    ]
)

print(encoded_df.head())