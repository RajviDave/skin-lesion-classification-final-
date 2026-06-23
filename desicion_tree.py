import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from simple_model import encoded_dataframe
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree


X=pd.DataFrame()
X[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]=encoded_dataframe[['Pigment_Encoded','Dots_Encoded','Streaks_Encoded','Regression_Encoded','Veil_Encoded']]
# print(X)

Y=pd.DataFrame()
Y[['Classes']]=encoded_dataframe[['Classes']]
# print(len(Y[['Classes']]))
# print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=50)
# print(len(X_train))
# print(len(Y_train))
# print(len(X_test))
# print(len(Y_test))

#train desicion tree using gini
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=50,max_depth=5,min_samples_leaf=5)
clf_gini.fit(X_train,Y_train)
# y_pred = clf_gini.predict(X_test)
# print(y_pred)

# accuracy = accuracy_score(Y_test, y_pred)
# print("Accuracy:", accuracy)
# print(confusion_matrix(Y_test, y_pred))

# Predictions on training data
y_train_pred = clf_gini.predict(X_train)

# Predictions on testing data
y_test_pred = clf_gini.predict(X_test)

train_acc = accuracy_score(Y_train, y_train_pred)
test_acc = accuracy_score(Y_test, y_test_pred)

# print("Training Accuracy:", train_acc)
# print("Testing Accuracy :", test_acc)

# print("Tree Depth:", clf_gini.get_depth())
# print("Leaves:", clf_gini.get_n_leaves())

scores = cross_val_score(clf_gini, X, Y, cv=5)

# print(scores)
# print("Mean Accuracy:", scores.mean())

train_pred = clf_gini.predict(X_train)
test_pred = clf_gini.predict(X_test)

print("Train Accuracy:", accuracy_score(Y_train, train_pred))
print("Test Accuracy:", accuracy_score(Y_test, test_pred))

# plt.figure(figsize=(15,8))
# plot_tree(
#     clf_gini,
#     feature_names=X.columns,
#     class_names=['Common Nevus','Atypical Nevus','Melanoma'],  # replace with actual class names
#     filled=True,
#     rounded=True
# )
# plt.show()