from data_loader import load_dataset
from dual_svm import DualSVM
from plot_utils import plot_decision_boundary
from kernels import linear_kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загружаем данные
X_train, X_test, y_train, y_test = load_dataset()

# Масштабирование (стандартное)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Кастомная реализация
svm_custom = DualSVM(C=1.0, kernel=linear_kernel)
svm_custom.fit(X_train, y_train)
y_pred_custom = svm_custom.predict(X_test)
print("Custom SVM Accuracy:", accuracy_score(y_test, y_pred_custom))

# sklearn SVM
svm_sklearn = SVC(C=1.0, kernel='linear')
svm_sklearn.fit(X_train, y_train)
y_pred_sklearn = svm_sklearn.predict(X_test)
print("Sklearn SVM Accuracy:", accuracy_score(y_test, y_pred_sklearn))

# Визуализация
plot_decision_boundary(svm_custom, X_train, y_train, kernel_type="Linear (Custom)")
plot_decision_boundary(svm_sklearn, X_train, y_train, kernel_type="Linear (Sklearn)")

