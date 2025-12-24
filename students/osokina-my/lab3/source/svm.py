import numpy as np
from scipy.optimize import minimize
from .kernels import LinearKernel


class SVM:
    def __init__(self, C=1.0, kernel=None, threshold=1e-5):
        self.C = C
        self.kernel = kernel if kernel is not None else LinearKernel()
        self.threshold = threshold

        # Параметры модели (заполняются после fit)
        self.alpha = None  # Множители Лагранжа
        self.support_vectors = None  # Опорные векторы
        self.support_labels = None  # Метки опорных векторов  
        self.support_alpha = None  # α для опорных векторов
        self.w0 = None  # Смещение

    def _compute_kernel_matrix(self, X):
        """Вычисление матрицы Грама K[i,j] = K(x_i, x_j)"""
        return self.kernel.compute(X, X)

    @staticmethod
    def _objective(alpha, Q):
        """Целевая функция: минимизируем −L(α) = −Σα_i + 0.5 * α^T Q α"""
        return -np.sum(alpha) + 0.5 * np.dot(alpha, np.dot(Q, alpha))

    def _compute_w0(self, X, y):
        """
        Вычисление смещения w0.
        
        Формула для граничного опорного вектора xₛ (где 0 < αₛ < C):
            w₀ = yₛ - Σⱼ αⱼyⱼK(xⱼ, xₛ)
        
        Для численной стабильности берём среднее по всем граничным опорным векторам:
            w₀ = (1/|M|) Σₛ∈M [yₛ - Σⱼ αⱼyⱼK(xⱼ, xₛ)]
        где M — множество граничных опорных векторов.
        
        Если граничных нет, используем все опорные векторы.
        """
        # Граничные опорные векторы: 0 < α < C (лежат точно на границе margin)
        margin_sv_mask = (self.alpha > self.threshold) & (self.alpha < self.C - self.threshold)

        if np.any(margin_sv_mask):
            margin_sv = X[margin_sv_mask]
            margin_labels = y[margin_sv_mask]
            K_margin = self.kernel.compute(self.support_vectors, margin_sv)
            w0_values = margin_labels - np.dot(self.support_alpha * self.support_labels, K_margin)
            self.w0 = np.mean(w0_values)
        else:
            # Fallback: используем все опорные векторы
            K = self.kernel.compute(self.support_vectors, self.support_vectors)
            self.w0 = np.mean(self.support_labels - np.dot(self.support_alpha * self.support_labels, K))

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Вычисляем матрицу Q один раз для оптимизации производительности
        K = self._compute_kernel_matrix(X)
        Q = np.outer(y, y) * K

        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}  # Σ α_i y_i = 0
        bounds = [(0, self.C) for _ in range(n_samples)]  # 0 ≤ α_i ≤ C
        alpha0 = np.zeros(n_samples)

        # Решаем задачу квадратичного программирования
        result = minimize(
            self._objective,
            alpha0,
            args=(Q,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        self.alpha = result.x

        # Находим опорные векторы (α > threshold)
        sv_mask = self.alpha > self.threshold
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.support_alpha = self.alpha[sv_mask]

        # Вычисляем w0 по граничным опорным векторам (0 < α < C)
        self._compute_w0(X, y)

    def predict(self, X):
        """Вычисление f(x) = Σ α_i y_i K(x_i, x) + w0"""
        K = self.kernel.compute(self.support_vectors, X)
        decision = np.dot(self.support_alpha * self.support_labels, K) + self.w0
        return np.where(decision >= 0, 1, -1).astype(int)

    @property
    def n_support_vectors(self):
        """Количество опорных векторов."""
        return len(self.support_vectors) if self.support_vectors is not None else 0
