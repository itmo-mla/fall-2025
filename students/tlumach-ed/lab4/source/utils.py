import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_diabetes


def load_regression_dataset(normalize=True):
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    if normalize:
        # простая стандартизация по признакам
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y, feature_names


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_scree(singular_values, savepath=None):
    # singular_values: array of s_j (sqrt(lambda_j))
    variances = singular_values ** 2
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(variances)+1), variances, marker='o', label='Eigenvalues (s^2)')
    plt.title('Scree plot')
    plt.xlabel('Component index')
    plt.ylabel('Eigenvalue (variance)')
    plt.grid(True)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def plot_cumulative_variance(explained_variance_ratio, savepath=None):
    cum = explained_variance_ratio.cumsum()
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(cum)+1), cum, marker='o', label='Cumulative explained variance')
    plt.hlines([0.9, 0.95], 1, len(cum), linestyles='dashed', label='90% / 95% thresholds')
    plt.title('Cumulative explained variance')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance ratio')
    plt.grid(True)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def plot_reconstruction_errors(ms, errors, savepath=None):
    plt.figure(figsize=(8,4))
    plt.plot(ms, errors, marker='o', label='Reconstruction MSE')
    plt.title('Reconstruction error vs number of components')
    plt.xlabel('m (components)')
    plt.ylabel('Mean squared reconstruction error')
    plt.grid(True)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()