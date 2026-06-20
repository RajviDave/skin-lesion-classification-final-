import pandas as pd
import numpy as np
from preprocessing import data
from preprocessing import C_N
# print(C_N)
print(data.columns)

data["Pigment Network\n(AT/T)"]=np.where(data["Pigment Network\n(AT/T)"]=="AT",0,1)
# print(data["Pigment Network\n(AT/T)"])

data["Streaks\n(A/P)"]=np.where(data["Streaks\n(A/P)"]=="A",0,1)
# print(data["Streaks\n(A/P)"])

data["Regression Areas\n(A/P)"]=np.where(data["Regression Areas\n(A/P)"]=="A",0,1)
# print(data["Regression Areas\n(A/P)"])

data["Blue-Whitish Veil\n(A/P)"]=np.where(data["Blue-Whitish Veil\n(A/P)"]=="A",0,1)
# print(data["Blue-Whitish Veil\n(A/P)"])