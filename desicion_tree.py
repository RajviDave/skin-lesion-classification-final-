import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from simple_model import encoded_dataframe

X=pd.DataFrame()
X[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]=encoded_dataframe[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]
# print(X)

Y=pd.DataFrame()
Y[['Classes']]=encoded_dataframe[['Classes']]
# print(len(Y[['Classes']]))
# print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=100)
# print(len(X_train))
# print(len(Y_train))
# print(len(X_test))
# print(len(Y_test))

#train desicion tree using gini
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=4,min_samples_leaf=3)
clf_gini.fit(X_train,Y_train)
y_pred = clf_gini.predict(X_test)
print(y_pred)