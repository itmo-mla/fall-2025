import logging

import numpy as np
import matplotlib.pyplot as plt

def plot_explained_variance(
        explained_variance_ratio,
        n_components_eff,
        model_name,
        save_path
):
    cumulative = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(8, 5))

    # Вклад отдельных компонент
    plt.bar(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
        alpha=0.6,
        label="Вклад отдельной компоненты"
    )

    # Накопленная дисперсия — гладкая линия
    plt.plot(
        range(1, len(explained_variance_ratio) + 1),
        cumulative,
        marker="o",
        linewidth=2,
        color="tab:orange",
        label="Накопленная объяснённая дисперсия"
    )

    # Эффективная размерность
    plt.axvline(
        n_components_eff,
        linestyle="--",
        linewidth=2,
        label=f"Эффективная размерность = {n_components_eff}"
    )

    plt.xlabel("Номер главной компоненты")
    plt.ylabel("Доля объяснённой дисперсии")
    plt.title(f"Определение эффективной размерности {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def compare_components(
        pca_custom, 
        pca_ref, 
        X, 
        n_samples=5, 
        n_components=5
):
    idx = np.random.choice(len(X), n_samples, replace=False)

    X_custom = pca_custom.transform(X[idx], n_components)
    X_ref = pca_ref.transform(X[idx])[:, :n_components]

    logging.info("Сравнение проекций (OwnPCA vs Sklearn PCA):")
    for i in range(n_samples):
        logging.info(f"Объект {i + 1}:")
        logging.info(f"OwnPCA: \n{np.round(X_custom[i], 9)}")
        logging.info(f"Sklearn PCA: \n{np.round(X_ref[i], 9)}")