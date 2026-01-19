from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data_load import load_dataset
from model import LinearKernel, PolynomialKernel, RBFKernel, SVM
from module import (
    compare_results,
    show_pairs,
    show_dist,
    visualize_model,
    plot_confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

def run_eda(df):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    show_pairs(df, ARTIFACTS_DIR / "pairplot_leaveornot.png")
    show_dist(df, ARTIFACTS_DIR / "target_distribution.png")


def run_experiments(X_train, X_test, y_train, y_test):
    kernels = {
        "linear": LinearKernel(),
        "poly": PolynomialKernel(degree=3),
        "rbf": RBFKernel(gamma=1.0),
    }

    my_scores = {}
    sklearn_scores = {}
    my_models = {}
    sklearn_models = {}

    for name, kernel in kernels.items():
        svm = SVM(kernel=kernel, C=1.0)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        my_scores[name] = acc
        my_models[name] = svm
        print(f"Custom SVM ({name}) accuracy: {acc:.4f}")

        visualize_model(
            svm,
            X_train[:500],
            y_train[:500],
            ARTIFACTS_DIR / f"boundary_{name}.png",
            kernel_name=name,
        )

        plot_confusion_matrix(
            y_test,
            y_pred,
            ARTIFACTS_DIR / f"confusion_matrix_{name}.png",
            kernel_name=name,
        )

    svc_models = {
        "linear": SVC(kernel="linear", C=1.0, random_state=42),
        "poly": SVC(kernel="poly", degree=3, C=1.0, random_state=42),
        "rbf": SVC(kernel="rbf", gamma=1.0, C=1.0, random_state=42),
    }

    for name, svc in svc_models.items():
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sklearn_scores[name] = acc
        sklearn_models[name] = svc
        print(f"sklearn SVC ({name}) accuracy: {acc:.4f}")

    compare_results(
        my_scores,
        sklearn_scores,
        ARTIFACTS_DIR / "accuracy_comparison.png",
    )

    return my_scores, sklearn_scores


def main():
    X_train, X_test, y_train, y_test, df = load_dataset()

    run_eda(df)

    my_scores, sklearn_scores = run_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n=== Summary ===")
    for k in my_scores:
        print(
            f"Kernel={k:6s} | custom SVM: {my_scores[k]:.4f} | "
            f"sklearn SVC: {sklearn_scores[k]:.4f}"
        )


if __name__ == "__main__":
    main()


