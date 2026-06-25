from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
from sklearn.model_selection import cross_val_score
from decision_tree import X_test,X_train,Y_test,Y_train

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=20,      # number of trees
    criterion='gini',
    max_depth=3,           # can also try None
    min_samples_leaf=3,
    random_state=100
)

# Train
rf_model.fit(X_train, Y_train.values.ravel())

# Predict
y_pred = rf_model.predict(X_test)#test accuracy
y_train_pred=rf_model.predict(X_train)#train accuracy

# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Test Accuracy:", accuracy)

test_accuracy=accuracy_score(Y_train,y_train_pred)
print("Training accuracy:",test_accuracy)

f1 = f1_score(Y_test, y_pred, average='weighted')
print("Weighted F1 Score:", f1)