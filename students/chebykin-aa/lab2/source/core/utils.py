from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any
import logging
import time

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def gaussian_kernel(
    x: np.ndarray
)-> np.ndarray:
    "Вычисляет ядро Гаусса"
    return np.exp(-2 * x**2)

def triangle_kernel(
    x: np.ndarray,
    eps = 1e-8
)-> np.ndarray:
    "Вычисляет треугольное ядро"
    return np.maximum(1 - np.abs(x), eps)

def quadratic_kernel(
    x: np.ndarray,
    eps = 1e-8
)-> np.ndarray:
    "Вычисляет квадратичное ядро"
    return np.maximum(1 - x**2, eps)

def minkowski_dist(
    X: np.ndarray, 
    y: np.ndarray,
    p: int
)-> np.ndarray:
    "Вычисляет расстояние минковского между новым объектом y и каждым старым объектом из X"
    return np.sum(np.abs(X - y) ** p, axis=1) ** (1/p)

def get_cross_validation_splits(
    n_examples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
)-> tuple[np.ndarray, np.ndarray]:
    """Проводит разбиение на подвыборки для кросс-валидации"""
    # Разобьем данные на n_splits частей
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_examples) if shuffle else np.arange(n_examples)
    split_indices = np.array_split(indices, n_splits)
    # Получим тренировочные/тестовые выборки
    folds = []
    for i in range(n_splits):
        val_idx = split_indices[i]
        train_idx = np.concatenate([split_indices[j] for j in range(n_splits) if j != i])
        folds.append((train_idx, val_idx))

    return folds

def compute_CCV_error(
    X: np.ndarray, 
    y: np.ndarray, 
    omega: list[int], 
    model: Any
):
    """Подсчитывает ошибку CCV для датасета"""
    errors = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        # Если элемент в omega, то исключаем его из тренировочной выборки
        if i in omega:
            train_idx = [j for j in omega if j != i]
        else:
            train_idx = omega

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[i].reshape(1, -1))[0]
        errors[i] = int(y_pred != y[i])

    return errors.mean()

def greedy_prototype_selection(
    X: np.ndarray, 
    y: np.ndarray, 
    model: Any,
    eps = 1e-6
):
    """Удаляет не-эталоны из датасета, используя жадную стратегию"""
    # Изначально omega - весь датасет
    omega = list(range(len(X)))
    noise, neutral = [], []
    # Считаем изначальную ошибку
    prev_CCV = compute_CCV_error(X, y, omega, model)

    # Итерируемся, пока CCV либо уменьшается, либо почти не увеличивается
    with ThreadPoolExecutor(max_workers=2) as executor:   
        while True:
            start = time.perf_counter()

            # Для каждого элемента из omega определяем, является он эталоном или нет
            new_omegas = [omega[:pos] + omega[pos+1:] for pos in range(len(omega))]    
            ccv_values = list(    
                executor.map(    
                    lambda om: compute_CCV_error(X, y, om, deepcopy(model)),    
                    new_omegas    
                )    
            )    

            # Находим лучшего кандидата
            best_pos = int(min(range(len(ccv_values)), key=lambda i: ccv_values[i]))    
            best_CCV = ccv_values[best_pos]    
            best_val = omega[best_pos]

            # Удаляем кандидата, если он не является эталоном
            if best_CCV <= prev_CCV + eps:
                if best_CCV < prev_CCV - eps:    
                    noise.append(best_val)    
                else:    
                    neutral.append(best_val)    

                omega.remove(best_val)
                prev_CCV = best_CCV    
            else:    
                break    

            end = time.perf_counter()
            logging.info(f"размер множества omega: {len(omega)}, время итерации: {end - start}")

    return omega, noise, neutral

def draw_q_plot(
    q_vals: dict,
    plot_path: str
):
    """Отрисовывает график эмпирического риска"""
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(q_vals.keys(), q_vals.values(), linestyle='-', color='red')
    axes.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes.set_xlabel('Число соседей')
    axes.set_ylabel('Частота ошибок')
    axes.set_title('Зависимость LOO от числа соседей')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def draw_selection_results(
    X: np.ndarray, 
    y: np.ndarray, 
    omega: list[int], 
    noise: list[int], 
    neutral: list[int], 
    plot_path: str
):
    """Отрисовывает результат работы алгоритма жадного удаления не-эталонных объектов"""
    # Уменьшаем размерность, чтобы отобразить точки на графике
    X2d = PCA(n_components=2).fit_transform(X)

    # Определяем категории объектов
    categories = {
        "Эталоны'": {"idx": omega, "marker": "o", "size": 80},
        "Нейтральные объекты": {"idx": neutral, "marker": "s", "size": 70},
        "Шумовые объекты": {"idx": noise, "marker": "x", "size": 90},
    }

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    for name, cfg in categories.items():
        if len(cfg["idx"]) == 0:
            continue
        axes.scatter(
            X2d[cfg["idx"], 0], X2d[cfg["idx"], 1],
            c=y[cfg["idx"]],
            cmap="tab10",
            marker=cfg["marker"],
            s=cfg["size"],
            label=name,
            edgecolor="k" if cfg["marker"] != "x" else None,
            linewidth=0.6
        )
    axes.set_title("Результат работы алгоритма жадного удаления не-эталонных объектов")
    axes.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

def calculate_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    file_path: str
):
    """Вычисляет качество предсказания моделей"""
    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)

    metrics_str = (
        f"Accuracy: {accuracy:.5f}\n"
        f"Precision: {precision:.5f}\n"
        f"Recall: {recall:.5f}\n"
        f"F1 Score: {f1:.5f}\n"
        f"\n"
    )
    with open(file_path, "a") as file:
        file.write(metrics_str)
