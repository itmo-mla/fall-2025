import matplotlib.pyplot as plt
import numpy as np
from svm import DualSVM

def plot_decision_2d(model: DualSVM, X, y, title="SVM decision boundary"):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    y_pm = np.where(y > 0, 1, -1)

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contour(xx, yy, zz, levels=[0], linewidths=2)
    plt.contour(xx, yy, zz, levels=[-1, 1], linestyles="--", linewidths=1)

    plt.scatter(X[y_pm == 1, 0], X[y_pm == 1, 1], s=25, marker="o", label="+1")
    plt.scatter(X[y_pm == -1, 0], X[y_pm == -1, 1], s=25, marker="s", label="-1")

    if hasattr(model, "sv_idx_") and model.sv_idx_ is not None and model.sv_idx_.size > 0:
        plt.scatter(X[model.sv_idx_, 0], X[model.sv_idx_, 1],
                    s=90, facecolors="none", edgecolors="k", linewidths=1.5, label="SV")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
