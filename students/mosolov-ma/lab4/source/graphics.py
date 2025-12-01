import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_scatter(principal_components, target, title='Scatter Plot главных компонент (PCA)'):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=target, cmap='RdBu', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Главная компонента 1 (PC1)')
    ax.set_ylabel('Главная компонента 2 (PC2)')
    ax.grid(True)
    return fig, ax

def plot_Em(singular_values, title='График ошибки реконструкции E'):
        lambdas = singular_values ** 2
        
        n = len(lambdas)

        m_vals = np.arange(1, n + 1)

        total = np.sum(lambdas)

        Em = np.array([np.sum(lambdas[m:]) / total for m in m_vals])

        ylabel = 'Ошибка реконструкции Eₘ'
        title = 'PCA: Ошибка реконструкции (Eₘ)'


        plt.figure(figsize=(8, 5))
        plt.plot(m_vals, Em, marker='o', linestyle='-')
        plt.xlabel('Количество главных компонент (m)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()