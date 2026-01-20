import numpy as np
import matplotlib.pyplot as plt


def plot_explained_variance(pca, X):
    explained_variance = pca.explained_variance(X)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

def plot_pca_scatter(pca, X, y):
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot')
    plt.colorbar(scatter, label='Class Label')
    plt.grid()
    plt.show()

