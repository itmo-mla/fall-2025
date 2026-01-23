import numpy as np
from sklearn.datasets import make_blobs, make_circles


def generate_linearly_separable_data(n_samples=200, n_features=2, random_state=42):
    """
    Генерирует линейно разделимые синтетические данные
    
    Args:
        n_samples: количество объектов
        n_features: количество признаков (по умолчанию 2 для визуализации)
        random_state: seed для воспроизводимости
    
    Returns:
        X: матрица признаков (n_samples, n_features)
        y: метки классов (n_samples,) в формате {-1, +1}
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        n_features=n_features,
        random_state=random_state,
        cluster_std=0.7,
        center_box=(-5, 5)
    )
    # Преобразуем метки в {-1, +1}
    y = np.where(y == 0, -1, 1)
    return X, y


def generate_circular_data(n_samples=200, noise=0.1, factor=0.5, random_state=42):
    """
    Генерирует круговые (нелинейно разделимые) синтетические данные
    Идеально для демонстрации работы RBF ядра
    
    Args:
        n_samples: количество объектов
        noise: уровень шума
        factor: масштаб между внутренним и внешним кругами
        random_state: seed для воспроизводимости
    
    Returns:
        X: матрица признаков (n_samples, 2)
        y: метки классов (n_samples,) в формате {-1, +1}
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )
    # Преобразуем метки в {-1, +1}
    y = np.where(y == 0, -1, 1)
    return X, y


def generate_xor_data(n_samples=200, noise=0.1, random_state=42):
    """
    Генерирует XOR-подобные данные (сложный нелинейный случай)
    
    Args:
        n_samples: количество объектов
        noise: уровень шума
        random_state: seed для воспроизводимости
    
    Returns:
        X: матрица признаков (n_samples, 2)
        y: метки классов (n_samples,) в формате {-1, +1}
    """
    np.random.seed(random_state)
    
    # Создаем 4 кластера в углах квадрата
    n_per_cluster = n_samples // 4
    
    # Верхний правый и нижний левый - класс +1
    X1 = np.random.randn(n_per_cluster, 2) * noise + [1, 1]
    X2 = np.random.randn(n_per_cluster, 2) * noise + [-1, -1]
    
    # Верхний левый и нижний правый - класс -1
    X3 = np.random.randn(n_per_cluster, 2) * noise + [-1, 1]
    X4 = np.random.randn(n_per_cluster, 2) * noise + [1, -1]
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.array([1] * (2 * n_per_cluster) + [-1] * (2 * n_per_cluster))
    
    # Перемешиваем
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y
