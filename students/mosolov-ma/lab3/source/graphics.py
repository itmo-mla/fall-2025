import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def heatmap_corr(corr_matrix):
    sns.set_theme(rc={'figure.figsize': (15, 8)})
    sns.heatmap(corr_matrix, annot=True, linewidths=3, cbar=False)
    plt.title("Матрица Корреляции")
    plt.show()


def plot_pca_scatter(principal_components, target):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=target, cmap='RdBu', alpha=0.7)
    ax.set_title('Scatter Plot главных компонент (PCA)')
    ax.set_xlabel('Главная компонента 1 (PC1)')
    ax.set_ylabel('Главная компонента 2 (PC2)')
    ax.grid(True)
    return fig, ax


def plot_decision_boundaries(ax, pca, w, b, label, color, support_vectors=None, X_train=None, y_train=None):
    P = pca.components_.T
    w_pca = P.T @ w

    xmin, xmax = ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 100)

    if w_pca[1] != 0:
        y_vals = (-b - w_pca[0] * x_vals) / w_pca[1]
        ax.plot(x_vals, y_vals, color=color, label=label, linewidth=2)

        y_vals_plus = (-b + 1 - w_pca[0] * x_vals) / w_pca[1]
        ax.plot(x_vals, y_vals_plus, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        y_vals_minus = (-b - 1 - w_pca[0] * x_vals) / w_pca[1]
        ax.plot(x_vals, y_vals_minus, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    else:
        ax.axvline(x=-b / w_pca[0], color=color, label=label, linewidth=2)
        ax.axvline(x=(-b + 1) / w_pca[0], color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=(-b - 1) / w_pca[0], color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    if support_vectors is not None:
        sv_pca = pca.transform(support_vectors)
        ax.scatter(sv_pca[:, 0], sv_pca[:, 1], 
                  s=200, facecolors='none', edgecolors='black', 
                  linewidths=2, label='Опорные вектора')
