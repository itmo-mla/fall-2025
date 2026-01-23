from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_explained_variance_ratio(
    evr: np.ndarray,
    out_path: str,
    title: str = "Explained Variance Ratio by Principal Components",
) -> None:
    x = np.arange(1, len(evr) + 1)

    plt.figure()
    plt.plot(x, evr, marker="o", label="Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cumulative_explained_variance(
    evr: np.ndarray,
    threshold: float,
    out_path: str,
    title: str = "Cumulative Explained Variance",
) -> None:
    x = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    plt.figure()
    plt.plot(x, cum, marker="o", label="Cumulative explained variance")
    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reconstruction_error_curve(
    errors: np.ndarray,
    out_path: str,
    title: str = "Reconstruction MSE vs Number of Components",
) -> None:
    x = np.arange(1, len(errors) + 1)

    plt.figure()
    plt.plot(x, errors, marker="o", label="Reconstruction MSE")
    plt.xlabel("Number of components")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_cumulative_variance_comparison(
    evr_ours: np.ndarray,
    evr_ref: np.ndarray,
    out_path: str,
    title: str = "Cumulative Explained Variance: Our PCA vs sklearn",
) -> None:
    k = min(len(evr_ours), len(evr_ref))
    x = np.arange(1, k + 1)
    cum_ours = np.cumsum(evr_ours[:k])
    cum_ref = np.cumsum(evr_ref[:k])

    plt.figure()
    plt.plot(x, cum_ours, marker="o", label="Our PCA (SVD)")
    plt.plot(x, cum_ref, marker="s", label="sklearn PCA")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
