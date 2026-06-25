from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,f1_score
from sklearn.model_selection import cross_val_score
from decision_tree import X_train,Y_train,X_test,Y_test,X,Y

# Create SVM model
svm_model = SVC(
    kernel='poly',      # 'linear', 'poly', 'rbf'
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

train_pred = svm_model.predict(X_train)

train_acc = accuracy_score(Y_train, train_pred)
test_acc = accuracy_score(Y_test, y_pred)

print("Train Accuracy SVM:", train_acc)
print("Test Accuracy SVM:", test_acc)

f1 = f1_score(Y_test, y_pred, average='weighted')
print("Weighted F1 Score:", f1)
