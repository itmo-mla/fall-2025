import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(svm_model, X, y, title="SVM Decision Boundary"):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    # Опорные векторы
    plt.scatter(svm_model.X_sv[:, 0], svm_model.X_sv[:, 1],
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.predict(grid).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--')
    plt.title(title)
    plt.legend()
    plt.savefig(f"images/{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()