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

def plot_decision_boundaries(ax, pca, w, b, label, color):
    P = pca.components_.T
    w_pca = P.T @ w
    xmin, xmax = ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 100)
    if w_pca[1] != 0:
        y_vals = (-b - w_pca[0] * x_vals) / w_pca[1]
        ax.plot(x_vals, y_vals, color=color, label=label)
    else:
        ax.axvline(x=-b / w_pca[0], color=color, label=label)

def plot_Q_plot(Q_plot, epochs):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(epochs + 1), Q_plot)
    plt.xlabel('Epoch')
    plt.ylabel('Q')
    plt.title('Эмпирический риск по эпохам')
    plt.grid(True)
    plt.show()

def plot_margins(margins):
    sorted_margins = np.sort(margins)
    i = np.arange(len(sorted_margins))
    plt.figure(figsize=(10, 5))
    plt.plot(i, sorted_margins, color='blue')
    plt.fill_between(i, sorted_margins, 0, where=sorted_margins < 0, color='red', alpha=0.7)
    plt.fill_between(i, sorted_margins, 0, where=(sorted_margins >= 0) & (sorted_margins < 0.4), color='khaki', alpha=0.7)
    plt.fill_between(i, sorted_margins, 0, where=sorted_margins >= 0.4, color='limegreen', alpha=0.7)
    plt.xlabel('$i$')
    plt.ylabel('Margin')
    plt.title('Margin для обучающей выборки')
    plt.grid(True)
    plt.ylim([min(sorted_margins) - 0.1, max(sorted_margins) + 0.1])
    plt.show()
