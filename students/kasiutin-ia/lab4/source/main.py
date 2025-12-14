import numpy as np
from sklearn.datasets import make_regression
from scipy.linalg import svd, eigvals
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components: int | str, eps: float=0.4):
        self.n_components = n_components
        self.eps = eps

    
    def _select_n_components(self, eigenvalues: np.ndarray) -> int:
        sorted_components_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_components_idx]

        m = 1
        while eigenvalues[m+1:].sum() / eigenvalues.sum() > self.eps:
            m += 1
        
        return m
    
    def plot_explained_variance(self, data: np.ndarray):
        eigenvalues = eigvals(data.T @ data)
        sorted_components_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_components_idx]

        explained_variance = []
        for m in range(len(eigenvalues)):
            explained_variance.append(eigenvalues[m:].sum() / eigenvalues.sum())
    
        plt.plot(explained_variance)
        plt.grid()
        plt.xlabel("n components")
        plt.ylabel("explained variance")
        plt.show()


    def fit(self, data: np.ndarray) -> None:
        V, D, U_T = svd(data, full_matrices=False)

        self.V = V
        self.singular_values = D
        self.D = np.diag(D)
        self.U_T = U_T

        self.G = self.V @ self.D

        if self.n_components == "auto":
            eigenvalues = eigvals(data.T @ data)
            n_components = self._select_n_components(eigenvalues)
            self.G = self.G[:, :n_components]
        
        elif isinstance(self.n_components, int):
            self.G = self.G[:, :self.n_components]


    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.G
