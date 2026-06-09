from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from desicion_tree import X_test,X_train,Y_test,Y_train

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    criterion='gini',
    max_depth=4,           # can also try None
    min_samples_leaf=3,
    random_state=100
)

# Train
rf_model.fit(X_train, Y_train.values.ravel())

# Predict
y_pred = rf_model.predict(X_test)