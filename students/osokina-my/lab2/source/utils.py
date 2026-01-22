import math
import numpy as np
import matplotlib.pyplot as plt

def leave_one_out_cross_validation(model, X, y, k_values):
    """Оптимизация параметра k методом LOO"""
    loo_scores = []

    X = np.array(X)
    y = np.array(y)

    for k in k_values:
        total_errors = 0

        for i in range(len(X)):
            # Создаем маски для обучающей и тестовой выборок
            mask = np.ones(len(X), dtype=bool)
            mask[i] = False

            # Все элементы кроме i-ого
            X_train = X[mask] 
            y_train = y[mask]
            # i-ый элемент
            X_test = X[i:i + 1]
            y_true = y[i]

            # Обучаем модель
            model.fit(X_train, y_train)

            # Предсказываем
            y_pred = model.predict_variable_window(X_test, k)

            # Считаем ошибки
            if y_pred[0] != y_true:
                total_errors += 1

        loo_error = total_errors / len(X)
        loo_scores.append(loo_error)

    return loo_scores


def plot_results(k_values, loo_scores, title):
    """Визуализация результатов LOO"""
    best_idx = np.argmin(loo_scores)
    best_k = k_values[best_idx]
    best_loo_score = loo_scores[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, loo_scores, 'bo-', linewidth=2, markersize=8)
    plt.plot(best_k, best_loo_score, 'ro', markersize=12, label=f'Лучшее k={best_k}')
    plt.xlabel('k')
    plt.ylabel('LOO ошибка')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return best_k, best_loo_score


def euclidean_distance(x1, x2):
    """Евклидово расстояние"""
    return math.sqrt(sum((a - b)**2 for a, b in zip(x1, x2)))


def gaussian_kernel(r):
    """Гауссово ядро: K(r) = exp(-r²/2)"""
    return math.exp(-r ** 2 / 2)


def triangular_kernel(r):
    """Треугольное ядро: K(r) = (1 - |r|) для |r| <= 1, иначе 0"""
    if abs(r) <= 1:
        return 1 - abs(r)
    return 0


def epanechnikov_kernel(r):
    """Ядро Епанечникова: K(r) = 3/4 * (1 - r²) для |r| <= 1, иначе 0"""
    if abs(r) <= 1:
        return 0.75 * (1 - r ** 2)
    return 0


# Словарь доступных ядер
KERNELS = {
    'gaussian': gaussian_kernel,
    'triangular': triangular_kernel,
    'epanechnikov': epanechnikov_kernel
}
