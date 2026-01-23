from typing import Union, Optional, Any, Literal, Type
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from skdim.id import MLE

from sklearn.metrics import pairwise_distances

#==================================================================

__all__ = [
    "estimate_intrinsic_dim_upgrade",
    "estimate_intrinsic_dim_skidim"
]

#==================================================================

#========== MLE of Intrinsic Dimensionality ==========

#---- Моя ручная реализация

def estimate_intrinsic_dim_upgrade(X, k_min=2, k_max=6, plot=False, save_path: Path | str | None = None):
    palette = px.colors.qualitative.Plotly

    distances = pairwise_distances(X)
    n = X.shape[0]
    m_k_inv_list = []

    for k in range(k_min, k_max + 1):
        m_i_inv_list = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])[1:k+1]
            if len(sorted_dists) < k:
                continue
            m_k = sorted_dists[-1]
            log_ratios = np.log(m_k / sorted_dists[:-1])
            m_i_inv = sum(log_ratios) / (k - 1) # (k - 2) == unbiased
            m_i_inv_list.append(m_i_inv)

        m_k_inv = 1.0 / np.mean(m_i_inv_list)
        m_k_inv_list.append(m_k_inv)

    intrinsic_dim = np.mean(m_k_inv_list)

    if plot or save_path is not None:
        sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        sns.lineplot(x=range(k_min, k_max + 1), y=m_k_inv_list, 
                     linestyle='--', marker='o', color=palette[3], ax=ax, label='Dimension Estimate')
        ax.set_xlabel('k neighbors', fontsize=12)
        ax.set_ylabel('Dimension Estimate m_k', fontsize=12)
        title = (f"Dimension Estimate (mean) vs K Neighbors " + 
                 f"(intrinsic dim = {intrinsic_dim:.2f} ~ {int(np.around(intrinsic_dim))})")
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis='both', labelsize=9)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if plot and save_path is None:
            plt.show()
        plt.close(fig)
        
    sns.reset_defaults()

    return int(np.around(intrinsic_dim))

#---- Библиотечная реализация (при comb='mle' результаты совпадают, но библиотечная немного быстрее)

def estimate_intrinsic_dim_skidim(
    X,
    k_min=2,
    k_max=6,
    comb='mean',
    unbiased=False,
    plot=False,
    save_path: Path | str | None = None,
):
    ks = range(k_min, k_max + 1)
    dims = []

    for k in ks:
        mle = MLE(unbiased=unbiased)
        mle.fit(X, n_neighbors=k, comb=comb) # mean
        dims.append(mle.dimension_)

    intrinsic_dim = np.mean(dims)

    if plot or save_path is not None:
        sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        sns.lineplot(x=ks, y=dims, linestyle='--', marker='o', color='purple', ax=ax, label='Dimension Estimate')
        ax.set_xlabel('k neighbors', fontsize=12)
        ax.set_ylabel('Dimension Estimate m_k', fontsize=12)
        title = (f"Dimension Estimate ({comb}) vs K Neighbors " + 
                f"(intrinsic dim = {intrinsic_dim:.2f} ~ {int(np.around(intrinsic_dim))})")
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis='both', labelsize=9)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if plot and save_path is None:
            plt.show()
        plt.close(fig)
        sns.reset_defaults()

    return int(np.around(intrinsic_dim)) 

#==================================================================
