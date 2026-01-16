import logging
import random
import shutil
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from core.utils import get_cross_validation_splits, greedy_prototype_selection, \
                       draw_selection_results, draw_q_plot, calculate_metrics, \
                       gaussian_kernel, triangle_kernel, quadratic_kernel
from core.model import OwnKNeighborsClassifier

def run_pipeline(
        X,
        y,
        weights_func,
        weights_name,
        results_path,
        p_degree,
        seed

):
    logging.info(f"Запустили пайплайн с ядром {weights_name}!")
    # Разобьем данные на фолды
    folds = get_cross_validation_splits(
        n_examples = len(y),
        n_splits = len(y),
        random_state = seed
    )

    # Найдем оптимальное количество ближайших соседей с помощью LOO
    q_by_k = {}
    for k in range(1, 80):
        q = []
        model = OwnKNeighborsClassifier(
            k_neighbors = k,
            weights = weights_func,
            metric = "minkowski",
            p = p_degree
        )
        for train_idx, val_idx in folds:
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            q.append(np.mean((y_pred != y[val_idx]).astype(int)))

        q_by_k[k] = np.mean(q)

    # Отрисуем графики эмпирического риска для различных k
    draw_q_plot(q_by_k, plot_path = f"{results_path}/{weights_name}/q_plot.png")
    # Найдем лучший параметр k
    best_k, best_q = min(q_by_k.items(), key=lambda item: item[1])
    logging.info(f"Минимальный эмпирический риск = {best_q:.4f} достигается при k = {best_k}")

    # Создаем модели с лучшими параметрами
    own_model = OwnKNeighborsClassifier(
        k_neighbors = best_k,
        weights = weights_func,
        metric = "minkowski",
        p = p_degree
    )
    sklearn_model = KNeighborsClassifier(
        n_neighbors = best_k,
        weights = weights_func,
        metric = "minkowski",
        p = p_degree
    )

    # Сравним качество работы OwnKNeighborsClassifier и KNeighborsClassifier на тесте
    own_model_preds = []
    sklearn_model_preds = []
    labels = []
    for train_idx, val_idx in folds:
        labels.append(y[val_idx])

        sklearn_model.fit(X[train_idx], y[train_idx])
        sklearn_model_preds.append(sklearn_model.predict(X[val_idx]))

        own_model.fit(X[train_idx], y[train_idx])
        own_model_preds.append(sklearn_model.predict(X[val_idx]))

    calculate_metrics(
        np.array(own_model_preds),
        np.array(labels),
        file_path = f"{results_path}/{weights_name}/own_model_without_selection_metrics.txt"
    )
    calculate_metrics(
        np.array(sklearn_model_preds),
        np.array(labels),
        file_path = f"{results_path}/{weights_name}/sklearn_model_without_selection_metrics.txt"
    )

    # Проведем отбор эталонов
    omega, noise, neutral = greedy_prototype_selection(X, y, own_model)
    # Визуализируем результаты работы алгоритма отбора эталонов
    draw_selection_results(
        X, 
        y,
        omega,
        noise,
        neutral,
        plot_path = f"{results_path}/{weights_name}/selection_results.png"
    )

    # Разделим выборку на эталонные/неэталонные объекты
    non_tgt_idx = [i for i in range(len(y)) if i not in omega]
    X_train, y_train = X[omega], y[omega]
    X_test, y_test = X[non_tgt_idx], y[non_tgt_idx]

    # Обучим модели только на эталонных объектах
    own_model.fit(X_train, y_train)
    sklearn_model.fit(X_train, y_train)

    # Сравним качество работы KNN с и без отбора эталонов
    own_model_preds = own_model.predict(X_test)
    sklearn_model_preds = sklearn_model.predict(X_test)
    calculate_metrics(
        own_model_preds,
        y_test,
        file_path = f"{results_path}/{weights_name}/own_model_with_selection_metrics.txt"
    )
    calculate_metrics(
        sklearn_model_preds,
        y_test,
        file_path = f"{results_path}/{weights_name}/sklearn_model_with_selection_metrics.txt"
    )

def main():
    # Зафиксируем seed и другие гиперпараметры
    P_DEGREE = 2
    SEED = 42
    kernels = [gaussian_kernel, triangle_kernel, quadratic_kernel]
    names = ["gaussian", "triangle", "quadratic"]
    np.random.seed(SEED)
    random.seed(SEED)

    # Создадим директорию для хранения результатов
    results_path = f"{os.getcwd()}/students/chebykin-aa/lab2/source/results"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(f"{results_path}/gaussian", exist_ok=True)
    os.makedirs(f"{results_path}/triangle", exist_ok=True)
    os.makedirs(f"{results_path}/quadratic", exist_ok=True)

    # Настроим логи
    logging.basicConfig(
        filename=f"{results_path}/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Получим датасет 
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns = data.feature_names)
    logging.info(f"\n{X.head()}")
    logging.info(f"\n{X.describe()}")
    # Получим метки
    y = pd.Series(data.target)
    logging.info("Статистика меток:")
    logging.info(f'Уникальные значения: {y.unique()}')
    logging.info(f'Распределение по классам:\n{y.value_counts()}')
    # Отнормируем признаки в [0; 1]
    scaler = MinMaxScaler(feature_range = (0, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns, index = X.index)
    logging.info(f"\n{X.describe()}")

    # Сконвертируем данные в необходимый формат
    X = X.to_numpy()
    y = y.to_numpy()

    for w_func, w_func_name in zip(kernels, names):
        run_pipeline(X, y, w_func, w_func_name, results_path, P_DEGREE, SEED)

if __name__ == "__main__":
    main()