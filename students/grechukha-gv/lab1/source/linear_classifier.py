import numpy as np

from margins import calculate_margin


def initialize_weights(n_features, method='random', X=None, y=None):
    """
    Инициализирует веса.
    Доступные методы: 
    - 'random': маленькие случайные значения
    - 'correlation': на основе корреляции с целевой переменной (требует X и y)
    """
    if method == 'random':
        return np.random.normal(scale=0.01, size=n_features)
    elif method == 'correlation':
        # Вычисляем корреляцию каждого признака с целевой переменной
        correlations = []
        for i in range(n_features):
            if np.std(X[:, i]) > 0:  # избегаем деления на 0
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        correlations = np.array(correlations)
        # Нормализуем и масштабируем
        weights = correlations * 0.1  # масштабируем для маленьких значений
        return weights
    else:
        raise ValueError(f"Неизвестный метод инициализации: {method}")

def add_bias_term(X):
    """
    Добавляет столбец единиц к матрице признаков для учета свободного члена w0
    X - матрица признаков формы (n_samples, n_features)
    Возвращает матрицу с добавленным столбцом единиц формы (n_samples, n_features + 1)
    """
    # Создаем столбец единиц такой же длины, как количество примеров в X
    bias = np.ones((X.shape[0], 1))
    # Объединяем по горизонтали: [единицы, X]
    return np.hstack([bias, X])

def quadratic_loss(margin):
    """
    Квадратичная функция потерь для одного объекта
    L(margin) = (1 - margin)² если margin < 1, иначе 0
    """
    if margin < 1:
        return (1 - margin) ** 2
    else:
        return 0

def quadratic_loss_gradient(w, x_i, y_i):
    """
    Вычисляет градиент квадратичной функции потерь для одного объекта
    Возвращает вектор градиента (той же размерности, что и w)
    """
    margin = calculate_margin(w, x_i, y_i)
    
    if margin < 1:
        # Градиент = 2 * (1 - margin) * (-y_i * x_i)
        return 2 * (1 - margin) * (-y_i * x_i)
    else:
        return np.zeros_like(w)  # нулевой градиент, если отступ >= 1

def logistic_loss(margin):
    """
    Логистическая функция потерь для одного объекта
    L(margin) = log(1 + exp(-margin))
    """
    # ограничиваем отступ для численной устойчивости
    safe_margin = np.clip(margin, -500, 500)
    # exp(-M)
    exp_neg_m = np.exp(-safe_margin)
    # 1 + exp(-M)
    one_plus_exp = 1.0 + exp_neg_m
    loss = np.log(one_plus_exp)
    return loss

def logistic_loss_gradient(w, x_i, y_i):
    """
    Градиент логистической потери по весам -σ(-M) * y * x
    где σ(-M) = 1 / (1 + exp(M)) - вероятность ошибки
    """
    margin = calculate_margin(w, x_i, y_i)
    # ограничиваем отступ для численной устойчивости
    safe_margin = np.clip(margin, -500, 500)

    # вычисляем σ(-M) = 1 / (1 + exp(M))
    # σ(-M) - это вероятность того, что модель ошибается
    sigma_neg_m = 1 / (1 + np.exp(safe_margin))

    # вычисляем градиент ∂L/∂w = -σ(-M) * y * x
    gradient = -sigma_neg_m * y_i * x_i

    return gradient

def logistic_loss_gradient_bias(margin, y_i):
    """
    Градиент логистической потери по смещению -σ(-M) * y
    """
    # ограничиваем отступ для численной устойчивости
    safe_margin = np.clip(margin, -500, 500)

    # вычисляем σ(-M) = 1 / (1 + exp(M))
    sigma_neg_m = 1 / (1 + np.exp(safe_margin))

    # вычисляем градиент ∂L/∂b = -σ(-M) * y
    gradient = -sigma_neg_m * y_i

    return gradient

def total_loss(w, X, y, loss_type='quadratic'):
    """
    Вычисляет общую функцию потерь на всей выборке
    (среднее значение потерь по всем объектам)
    """
    total = 0
    for i in range(len(y)):
        margin = calculate_margin(w, X[i], y[i])
        if loss_type == 'quadratic':
            total += quadratic_loss(margin)
        elif loss_type == 'logistic':
            total += logistic_loss(margin)
        else:
            raise ValueError(f"Неизвестный тип функции потерь: {loss_type}")
    return total / len(y)
