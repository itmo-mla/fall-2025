import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm import CustomSVM
from visualize import plot_decision_boundary
from loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


if __name__ == "__main__":
    X, y = load_dataset("wine")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    KERNEL = "poly"
    print("Обучение SVM")
    custom_svm = CustomSVM(C=1.0, kernel=KERNEL)
    custom_svm.fit(X, y)
    y_pred_custom = custom_svm.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    print(f"Custom SVM Accuracy:   {acc_custom:.3f}")

    print("Обучение эталонной реализации")
    sklearn_svm = SVC(kernel=KERNEL, C=1.0)
    sklearn_svm.fit(X, y)
    y_pred_sklearn = sklearn_svm.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn SVM Accuracy:  {acc_sklearn:.3f}")

    plot_decision_boundary(custom_svm, X, y, f"Custom {KERNEL} SVM")
    plot_decision_boundary(
        type('SklearnWrapper', (), {
            'X_sv': X[sklearn_svm.support_],
            'y_sv': y[sklearn_svm.support_],
            'predict': lambda self, X: sklearn_svm.predict(X)
        })(), X, y, f"Sklearn {KERNEL} SVM"
    )