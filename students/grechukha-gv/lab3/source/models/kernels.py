import numpy as np


def linear_kernel(X1, X2):
    """
    Линейное ядро: K(x_i, x_j) = x_i^T * x_j
    
    Args:
        X1: матрица размера (n1, d)
        X2: матрица размера (n2, d)
    
    Returns:
        Матрица Грама размера (n1, n2)
    """
    return np.dot(X1, X2.T)


def rbf_kernel(X1, X2, gamma=1.0):
    """
    RBF (Radial Basis Function) или гауссово ядро:
    K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
    
    Args:
        X1: матрица размера (n1, d)
        X2: матрица размера (n2, d)
        gamma: параметр ширины ядра (чем больше, тем уже)
    
    Returns:
        Матрица Грама размера (n1, n2)
    """
    # Вычисляем квадраты евклидовых расстояний
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)  # (n1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)  # (1, n2)
    distances_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)  # (n1, n2)
    
    # Применяем гауссову функцию
    return np.exp(-gamma * distances_sq)


def polynomial_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """
    Полиномиальное ядро: K(x_i, x_j) = (gamma * x_i^T * x_j + c)^d
    
    Args:
        X1: матрица размера (n1, d)
        X2: матрица размера (n2, d)
        degree: степень полинома d
        gamma: масштабирующий коэффициент (по умолчанию 1.0)
        coef0: свободный коэффициент c
    
    Returns:
        Матрица Грама размера (n1, n2)
    """
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree


def compute_kernel_matrix(X, kernel='linear', **kernel_params):
    """
    Вычисляет матрицу Грама для обучающей выборки
    
    Args:
        X: обучающая выборка размера (n, d)
        kernel: тип ядра ('linear', 'rbf', 'polynomial')
        **kernel_params: параметры ядра
    
    Returns:
        Матрица Грама размера (n, n)
    """
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'rbf':
        gamma = kernel_params.get('gamma', 1.0)
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == 'polynomial':
        degree = kernel_params.get('degree', 3)
        gamma = kernel_params.get('gamma', 1.0)
        coef0 = kernel_params.get('coef0', 1.0)
        K = polynomial_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Неизвестный тип ядра: {kernel}")
    
    return K


def compute_kernel_test(X_train, X_test, kernel='linear', **kernel_params):
    """
    Вычисляет матрицу ядра между обучающей и тестовой выборками
    
    Args:
        X_train: обучающая выборка размера (n_train, d)
        X_test: тестовая выборка размера (n_test, d)
        kernel: тип ядра ('linear', 'rbf', 'polynomial')
        **kernel_params: параметры ядра
    
    Returns:
        Матрица ядра размера (n_test, n_train)
    """
    if kernel == 'linear':
        K = linear_kernel(X_test, X_train)
    elif kernel == 'rbf':
        gamma = kernel_params.get('gamma', 1.0)
        K = rbf_kernel(X_test, X_train, gamma=gamma)
    elif kernel == 'polynomial':
        degree = kernel_params.get('degree', 3)
        gamma = kernel_params.get('gamma', 1.0)
        coef0 = kernel_params.get('coef0', 1.0)
        K = polynomial_kernel(X_test, X_train, degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Неизвестный тип ядра: {kernel}")
    
    return K
