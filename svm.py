from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from desicion_tree import X_train,Y_train,X_test,Y_test,X,Y

# Create SVM model
svm_model = SVC(
    kernel='rbf',      # 'linear', 'poly', 'rbf'
    C=1.0,
    gamma='scale',
    random_state=100
)

svm_model.fit(X_train, Y_train.values.ravel())
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Test Accuracy:", accuracy)

print(classification_report(Y_test, y_pred))

# Cross Validation
scores = cross_val_score(svm_model, X, Y.values.ravel(), cv=5)

print("CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())