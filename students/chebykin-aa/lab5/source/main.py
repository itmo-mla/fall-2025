import os
import shutil
import random
import logging

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from core.model_newton import OwnLogisticRegressionNewtonRafson
from core.model_irls import OwnLogisticRegressionIRLS
from core.utils import calculate_metrics, plot_probabilities


def main():
    # Зафиксируем seed и другие гиперпараметры
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # Создадим директорию для хранения результатов
    results_path = f"{os.getcwd()}/students/chebykin-aa/lab5/source/results"
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

    # Загружаем датасет Diabetes для классификации
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Конвертируем метки в {-1; 1},
    y = y.map({0: -1, 1: 1})
    logging.info("Статистика меток:")
    logging.info(f'Уникальные значения: {y.unique()}')
    logging.info(f'Распределение по классам:\n{y.value_counts()}')

    # Нормируем признаки
    X = (X - X.mean()) / X.std()

    # Разбиваем данные на выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size = 0.2,
        random_state = SEED
    )

    # Обучение и тестирование LR NewtonRafson
    model_newton = OwnLogisticRegressionNewtonRafson(max_iter=1000)
    model_newton.fit(X_train, y_train)

    preds_newton = model_newton.predict(X_test)
    probs_newton = model_newton.predict_proba(X_test)

    calculate_metrics(
        preds_newton, 
        y_test, 
        file_path = f"{results_path}/metrics.txt"
    )

    # Обучение и тестирование LR IRLS
    model_irls = OwnLogisticRegressionIRLS(max_iter=1000)
    model_irls.fit(X_train, y_train)

    preds_irls = model_irls.predict(X_test)
    probs_irls = model_irls.predict_proba(X_test)

    calculate_metrics(
        preds_irls, 
        y_test, 
        file_path = f"{results_path}/metrics.txt"
    )

    # Обучение и тестирование модели из sklearn
    model_sklearn = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000
    )
    model_sklearn.fit(X_train, y_train)

    preds_sklearn = model_sklearn.predict(X_test)
    probs_sklearn = model_sklearn.predict_proba(X_test)[:, 1]

    calculate_metrics(
        preds_sklearn, 
        y_test, 
        file_path = f"{results_path}/metrics.txt"
    )

    logging.info(
        f"Средние отличия вероятностей:\n"
        f"Newton vs IRLS: {np.mean(np.abs(probs_newton - probs_irls)):.6f}\n"
        f"Newton vs Sklearn: {np.mean(np.abs(probs_newton - probs_sklearn)):.6f}\n"
        f"IRLS vs Sklearn: {np.mean(np.abs(probs_irls - probs_sklearn)):.6f}"
    )

    plot_probabilities(
        probs_newton, 
        probs_irls, 
        probs_sklearn,
        save_path = f"{results_path}/probs.png"
    )


if __name__ == "__main__":
    main()
