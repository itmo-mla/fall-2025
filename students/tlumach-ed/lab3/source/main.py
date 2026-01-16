from data_loader import load_dataset
from dual_svm import DualSVM
from plot_utils import plot_decision_boundary
from kernels import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


DATASET_TYPE = "moons"           # "classification" | "moons"
KERNEL_TYPE = "rbf"              # "linear" | "poly" | "rbf"
C_VALUE = 1.0

X_train, X_test, y_train, y_test = load_dataset(dataset_type=DATASET_TYPE)

# Масштабирование
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if KERNEL_TYPE == "linear":
    custom_kernel = linear_kernel
    sklearn_kernel = "linear"

elif KERNEL_TYPE == "poly":
    custom_kernel = lambda x, y: polynomial_kernel(x, y, degree=3)
    sklearn_kernel = "poly"

elif KERNEL_TYPE == "rbf":
    custom_kernel = lambda x, y: rbf_kernel(x, y, gamma=0.5)
    sklearn_kernel = "rbf"

else:
    raise ValueError("Unknown kernel type")


svm_custom = DualSVM(C=C_VALUE, kernel=custom_kernel)
svm_custom.fit(X_train, y_train)

y_pred_custom = svm_custom.predict(X_test)
print("Custom SVM Accuracy:", accuracy_score(y_test, y_pred_custom))


svm_sklearn = SVC(
    C=C_VALUE,
    kernel=sklearn_kernel,
    degree=3,
    gamma=0.5
)

svm_sklearn.fit(X_train, y_train)
y_pred_sklearn = svm_sklearn.predict(X_test)
print("Sklearn SVM Accuracy:", accuracy_score(y_test, y_pred_sklearn))

plot_decision_boundary(svm_custom, X_train, y_train, kernel_type=f"{KERNEL_TYPE} (Custom)")
plot_decision_boundary(svm_sklearn, X_train, y_train, kernel_type=f"{KERNEL_TYPE} (Sklearn)")
