import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_diabetes_regression
from pca import PCASVD, PCAEigen, choose_effective_dim


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 0.0,
) -> np.ndarray:
    """
    Fit linear regression with optional L2 regularization using the closed-form solution.

    We add a bias term explicitly and do not regularize it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape
    X_design = np.hstack([X, np.ones((n_samples, 1))])

    I = np.eye(n_features + 1)
    I[-1, -1] = 0.0  # do not regularize bias

    A = X_design.T @ X_design + l2_reg * I
    b = X_design.T @ y

    w = np.linalg.solve(A, b)
    return w


def predict_linear_regression(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    X_design = np.hstack([X, np.ones((n_samples, 1))])
    return X_design @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compare_pca_implementations(X_train: np.ndarray) -> Tuple[PCASVD, PCAEigen]:
    """
    Fit PCA via SVD and via eigen-decomposition and print numerical differences.
    """
    pca_svd = PCASVD().fit(X_train)
    pca_eig = PCAEigen().fit(X_train)

    # Compare explained variances
    ev_diff = np.max(np.abs(pca_svd.explained_variance_ - pca_eig.explained_variance_))
    evr_diff = np.max(
        np.abs(pca_svd.explained_variance_ratio_ - pca_eig.explained_variance_ratio_)
    )

    # Compare components up to sign
    cosines = []
    for i in range(pca_svd.components_.shape[0]):
        v1 = pca_svd.components_[i]
        v2 = pca_eig.components_[i]
        cos = float(np.abs(np.dot(v1, v2)))
        cosines.append(cos)

    print("=== Equivalence of PCA implementations (SVD vs eigen) ===")
    print(f"Max |explained_variance difference|   : {ev_diff:.3e}")
    print(f"Max |explained_variance_ratio diff|  : {evr_diff:.3e}")
    print(f"Cosines between corresponding components (|dot|): {cosines}")
    print()

    return pca_svd, pca_eig


def plot_explained_variance(pca: PCASVD, threshold: float = 0.95) -> int:
    """
    Plot scree plot and cumulative explained variance; return effective dimension.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ratios = pca.explained_variance_ratio_
    n_components = len(ratios)
    idx = np.arange(1, n_components + 1)

    # Scree plot
    plt.figure(figsize=(6, 4))
    plt.plot(idx, ratios, marker="o")
    plt.xlabel("Номер компоненты")
    plt.ylabel("Доля объяснённой дисперсии")
    plt.title("Scree plot (собственные значения)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pca_scree_plot.png"))

    # Cumulative explained variance
    cumulative = np.cumsum(ratios)
    eff_dim = choose_effective_dim(ratios, threshold=threshold)

    plt.figure(figsize=(6, 4))
    plt.step(idx, cumulative, where="mid", label="Накопленная доля дисперсии")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Порог {threshold:.2f}")
    plt.axvline(eff_dim, color="green", linestyle="--", label=f"m = {eff_dim}")
    plt.xlabel("Число компонент")
    plt.ylabel("Накопленная доля дисперсии")
    plt.title("Накопленная объяснённая дисперсия")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pca_cumulative_variance.png"))

    return eff_dim


def plot_pc_scatter(Z: np.ndarray, y: np.ndarray) -> None:
    """
    2D scatter в пространстве первых двух главных компонент.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if Z.shape[1] < 2:
        return

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=20, edgecolor="k", linewidths=0.3)
    plt.colorbar(sc, label="Целевая переменная")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Данные в пространстве первых двух главных компонент")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pca_scatter_pc12.png"))


def run_experiment() -> None:
    # 1. Загрузка и предобработка данных
    data = load_diabetes_regression()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 2. Сравнение реализаций PCA
    pca_svd_full, pca_eig_full = compare_pca_implementations(X_train)

    # 3. Определение эффективной размерности
    threshold = 0.95
    eff_dim = plot_explained_variance(pca_svd_full, threshold=threshold)
    print(f"Эффективная размерность (порог {threshold:.2f} по доле дисперсии): m = {eff_dim}")

    # 4. Снижение размерности с помощью PCA (SVD)
    pca_svd = PCASVD(n_components=eff_dim).fit(X_train)
    Z_train = pca_svd.transform(X_train)
    Z_val = pca_svd.transform(X_val)
    Z_test = pca_svd.transform(X_test)

    plot_pc_scatter(Z_train, y_train)

    # 5. Линейная регрессия в исходном и PCA-пространстве
    l2_reg = 1e-2
    print(f"\n=== Линейная регрессия (L2 = {l2_reg}) ===")

    w_full = fit_linear_regression(X_train, y_train, l2_reg=l2_reg)
    y_val_pred_full = predict_linear_regression(X_val, w_full)
    y_test_pred_full = predict_linear_regression(X_test, w_full)

    w_pca = fit_linear_regression(Z_train, y_train, l2_reg=l2_reg)
    y_val_pred_pca = predict_linear_regression(Z_val, w_pca)
    y_test_pred_pca = predict_linear_regression(Z_test, w_pca)

    print("Исходное пространство признаков:")
    print(f"  Val MSE = {mse(y_val, y_val_pred_full):.4f}, R^2 = {r2_score(y_val, y_val_pred_full):.4f}")
    print(f"  Test MSE = {mse(y_test, y_test_pred_full):.4f}, R^2 = {r2_score(y_test, y_test_pred_full):.4f}")

    print(f"\nПространство первых {eff_dim} главных компонент:")
    print(f"  Val MSE = {mse(y_val, y_val_pred_pca):.4f}, R^2 = {r2_score(y_val, y_val_pred_pca):.4f}")
    print(f"  Test MSE = {mse(y_test, y_test_pred_pca):.4f}, R^2 = {r2_score(y_test, y_test_pred_pca):.4f}")


if __name__ == "__main__":
    run_experiment()

