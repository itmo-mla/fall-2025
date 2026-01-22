import kagglehub
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from model.logistic_regression import LogisticRegressionIRLS, NewtonRaphsonLogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data_load.data_load import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

def run_experiments(X_train, X_test, y_train, y_test):
    custom_model_irls = LogisticRegressionIRLS(h_t=1.0, iter=1000, tol=1e-6)
    custom_model_irls.fit(X_train, y_train)
    y_pred_custom = custom_model_irls.predict(X_test)
    acc_custom_irls = accuracy_score(y_test, y_pred_custom)

    custom_model_newton = NewtonRaphsonLogisticRegression(h_t=1.0, iter=1000, tol=1e-6)
    custom_model_newton.fit(X_train, y_train)
    y_pred_custom = custom_model_newton.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)

    sklearn_model = SklearnLogReg(max_iter=1000, solver="lbfgs")
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    return {
        "custom_accuracy_irls": acc_custom_irls,
        "newton_custom_accuracy": acc_custom,
        "sklearn_accuracy": acc_sklearn
    }

def run_eda(df: pd.DataFrame):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    stats_file = ARTIFACTS_DIR / "descriptive_stats.csv"
    df.describe().to_csv(stats_file)

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / f"{col}_hist.png")
        plt.close()

    corr = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_matrix.png")
    plt.close()

    corr_file = ARTIFACTS_DIR / "correlation_matrix.csv"
    corr.to_csv(corr_file)
    print(f"Correlation matrix saved to {corr_file}")

def main():
    X_train, X_test, y_train, y_test, df = load_dataset()

    run_eda(df)

    my_scores = run_experiments(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n=== Summary ===")
    print(
        f"Custom Logistic Regression IRLS: {my_scores["custom_accuracy_irls"]:.4f} \n"
        f"Custom Logistic Regression Newton: {my_scores["newton_custom_accuracy"]:.4f} \n"
        f"Sklearn Logistic Regression: {my_scores["sklearn_accuracy"]:.4f}"
    )

if __name__ == "__main__":
    main()