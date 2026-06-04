import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from simple_model import encoded_dataframe

X=pd.DataFrame()
X[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]=encoded_dataframe[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]
print(X)
