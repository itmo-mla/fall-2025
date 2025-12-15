from __future__ import annotations

from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data_load import load_dataset
from model import LinearKernel, PolynomialKernel, RBFKernel, SVM
from module import (
    plot_accuracy_comparison,
    plot_pairplot,
    plot_target_distribution,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def run_eda(df) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_pairplot(df, ARTIFACTS_DIR / "pairplot_leaveornot.png")
    plot_target_distribution(df, ARTIFACTS_DIR / "target_distribution.png")


def run_experiments(X_train, X_test, y_train, y_test):
    kernels = {
        "linear": LinearKernel(),
        "poly": PolynomialKernel(degree=3),
        "rbf": RBFKernel(gamma=1.0),
    }

    custom_scores = {}
    baseline_scores = {}

    for name, kernel in kernels.items():
        svm = SVM(kernel=kernel, C=1.0)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        custom_scores[name] = acc
        print(f"Custom SVM ({name}) accuracy: {acc:.4f}")

    svc_models = {
        "linear": SVC(kernel="linear", C=1.0, random_state=42),
        "poly": SVC(kernel="poly", degree=3, C=1.0, random_state=42),
        "rbf": SVC(kernel="rbf", gamma=1.0, C=1.0, random_state=42),
    }

    for name, svc in svc_models.items():
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        baseline_scores[name] = acc
        print(f"sklearn SVC ({name}) accuracy: {acc:.4f}")

    plot_accuracy_comparison(
        custom_scores,
        baseline_scores,
        ARTIFACTS_DIR / "accuracy_comparison.png",
    )

    return custom_scores, baseline_scores


def main():
    X_train, X_test, y_train, y_test, df = load_dataset()
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df["LeaveOrNot"].value_counts(normalize=True))

    run_eda(df)

    custom_scores, baseline_scores = run_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n=== Summary ===")
    for k in custom_scores:
        print(
            f"Kernel={k:6s} | custom SVM: {custom_scores[k]:.4f} | "
            f"sklearn SVC: {baseline_scores[k]:.4f}"
        )


if __name__ == "__main__":
    main()


