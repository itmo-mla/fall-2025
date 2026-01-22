import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from source.svm_dual import DualSVM


def _ensure_outputs():
    os.makedirs("outputs/figures", exist_ok=True)


def plot_pca_projection(X_train, y_train, X_test, y_test, title="PCA projection", filename="pca_projection.png"):
    _ensure_outputs()
    pca = PCA(n_components=2, random_state=42)
    Xtr2 = pca.fit_transform(X_train)
    Xte2 = pca.transform(X_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(Xtr2[:, 0], Xtr2[:, 1], c=y_train, cmap="bwr", alpha=0.55, label="train")
    plt.scatter(Xte2[:, 0], Xte2[:, 1], c=y_test, cmap="bwr", alpha=0.85, marker="x", label="test")
    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=160)
    plt.close()


def plot_decision_boundary_pca(model: DualSVM, X_train, y_train, title, filename, support_vectors=None):
    _ensure_outputs()
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_train)

    x_min, x_max = X2[:, 0].min() - 0.8, X2[:, 0].max() + 0.8
    y_min, y_max = X2[:, 1].min() - 0.8, X2[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    grid_orig = pca.inverse_transform(grid2)

    Z = model.predict(grid_orig).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="bwr")
    plt.contour(xx, yy, Z, levels=[0], colors="k", linestyles="--", linewidths=1.5)

    plt.scatter(X2[:, 0], X2[:, 1], c=y_train, cmap="bwr", alpha=0.7, label="train")

    if support_vectors is not None and len(support_vectors) > 0:
        sv2 = pca.transform(support_vectors)
        plt.scatter(sv2[:, 0], sv2[:, 1], s=90, facecolors="none", linewidths=1.5, label="SV")

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=160)
    plt.close()


def plot_decision_boundary_2d(model, X, y, title, filename, support_vectors=None):
    """
    For already 2D datasets (e.g. make_circles). model must implement predict(X) -> {-1, +1} or {0,1}.
    """
    _ensure_outputs()

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 320), np.linspace(y_min, y_max, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="bwr")
    plt.contour(xx, yy, Z, levels=[0], colors="k", linestyles="--", linewidths=1.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.8, label="data")

    if support_vectors is not None and len(support_vectors) > 0:
        sv = np.asarray(support_vectors)
        plt.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", linewidths=1.5, label="SV")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=160)
    plt.close()


def plot_confusion_matrices(y_true, y_pred_custom, y_pred_sklearn, title, filename):
    _ensure_outputs()

    cm_custom = confusion_matrix(y_true, y_pred_custom, labels=[-1, 1])
    cm_sklearn = confusion_matrix(y_true, y_pred_sklearn, labels=[-1, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ConfusionMatrixDisplay(cm_custom, display_labels=[-1, 1]).plot(ax=axes[0], values_format="d")
    axes[0].set_title("Custom")

    ConfusionMatrixDisplay(cm_sklearn, display_labels=[-1, 1]).plot(ax=axes[1], values_format="d")
    axes[1].set_title("sklearn")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{filename}", dpi=170)
    plt.close()
