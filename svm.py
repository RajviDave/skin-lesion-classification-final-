from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from desicion_tree import X_train,Y_train,X_test

# Create SVM model
svm_model = SVC(
    kernel='rbf',      # 'linear', 'poly', 'rbf'
    C=1.0,
    gamma='scale',
    random_state=100
)

svm_model.fit(X_train, Y_train.values.ravel())
y_pred = svm_model.predict(X_test)