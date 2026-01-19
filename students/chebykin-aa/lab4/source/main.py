import os
import shutil
import random
import logging

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

from core.model import OwnPCA
from core.utils import plot_explained_variance, compare_components

def main():
    # Зафиксируем seed и другие гиперпараметры
    SEED = 42
    VAR_THRESH = 0.95
    np.random.seed(SEED)
    random.seed(SEED)

    # Создадим директорию для хранения результатов
    results_path = f"{os.getcwd()}/students/chebykin-aa/lab4/source/results"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)
    print(f"{results_path}")

    # Настроим логи
    logging.basicConfig(
        filename=f"{results_path}/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Загружаем датасет Diabetes для задачи линейной регрессии
    data = load_diabetes(as_frame=True)
    X = data.data

    # Нормируем признаки
    X = (X - X.mean()) / X.std()
    X = X.values

    # Вычисляем главные компоненты
    pca = OwnPCA(variance_threshold = VAR_THRESH)
    pca.fit(X)

    sk_pca = PCA(n_components = X.shape[1])
    sk_pca.fit(X)

    # Рассчитываем эффективную размерность
    logging.info(f"Эффективная размерность (95% объяснённой дисперсии) для OwnPCA: {pca.n_components_eff_}")
    plot_explained_variance(
        pca.explained_variance_ratio_, 
        pca.n_components_eff_, 
        model_name = "OwnPCA",
        save_path=f"{results_path}/OwnPCA_effective_dim.png"
    )

    cum_var = np.cumsum(sk_pca.explained_variance_ratio_)
    sk_n_components_eff = np.searchsorted(cum_var, VAR_THRESH) + 1
    logging.info(
        f"Эффективная размерность (95% объяснённой дисперсии) для Sklearn PCA: "
        f"{sk_n_components_eff}"
    )
    plot_explained_variance(
        sk_pca.explained_variance_ratio_,
        sk_n_components_eff,
        model_name = "Sklearn PCA",
        save_path=f"{results_path}/SklearnPCA_effective_dim.png"
    )

    # Сравниваем компоненты для N случайных объектов
    N = 5
    compare_components(
        pca, sk_pca, X, n_samples=N, n_components=pca.n_components_eff_
    )


if __name__ == "__main__":
    main()
