import numpy as np
from scipy.optimize import minimize, LinearConstraint

from .kernels import compute_kernel_matrix, compute_kernel_test


class SVM:
    """
    Support Vector Machine через решение двойственной задачи
    
    Двойственная задача:
        max_λ Σ λ_i - 1/2 Σ_ij λ_i λ_j y_i y_j K(x_i, x_j)
    
    При ограничениях:
        0 ≤ λ_i ≤ C
        Σ λ_i y_i = 0
    """
    
    def __init__(self, C=1.0, kernel='linear', **kernel_params):
        """
        Args:
            C: параметр регуляризации (чем больше, тем жестче штраф за ошибки)
            kernel: тип ядра ('linear', 'rbf', 'polynomial')
            **kernel_params: параметры ядра (gamma для rbf, degree и coef0 для polynomial)
        """
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params
        
        self.lambdas = None  # множители Лагранжа
        self.support_vectors = None  # опорные векторы
        self.support_vector_labels = None  # метки опорных векторов
        self.support_vector_lambdas = None  # λ для опорных векторов
        self.support_vector_indices = None  # индексы опорных векторов в исходной выборке
        self.b = None  # bias
        self.X_train = None
        self.y_train = None
        
        # Кэш для оптимизации
        self._K_yy = None
        
    def _dual_objective(self, lambdas):
        """
        Целевая функция двойственной задачи (минимизируем отрицание)
        
        L(λ) = -Σ λ_i + 1/2 Σ_ij λ_i λ_j y_i y_j K(x_i, x_j)
        
        Args:
            lambdas: множители Лагранжа (n,)
        
        Returns:
            Значение целевой функции (для минимизации)
        """
        # Векторизованное вычисление с использованием кэша
        # 1/2 * λ^T * (y y^T ⊙ K) * λ - 1^T * λ
        term1 = 0.5 * np.dot(lambdas, np.dot(self._K_yy, lambdas))
        term2 = np.sum(lambdas)
        return term1 - term2
    
    def _dual_gradient(self, lambdas):
        """
        Градиент целевой функции
        
        ∇L(λ) = (y y^T ⊙ K) * λ - 1
        
        Args:
            lambdas: множители Лагранжа (n,)
        
        Returns:
            Градиент (n,)
        """
        return np.dot(self._K_yy, lambdas) - np.ones(len(lambdas))
    
    def _initialize_lambdas(self, y):
        """
        Умная инициализация λ с учётом пропорций классов
        
        Args:
            y: метки классов (n,)
        
        Returns:
            Начальные значения λ (n,)
        """
        n_samples = len(y)
        lambdas_init = np.zeros(n_samples)
        
        # Инициализируем с учётом пропорций классов
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            proportion = 1 - count / n_samples
            lambdas_init[y == label] = proportion * 0.1  # небольшое начальное значение
        
        return lambdas_init
    
    def fit(self, X, y, verbose=True):
        """
        Обучение SVM
        
        Args:
            X: обучающая выборка (n, d)
            y: метки классов (n,) в формате {-1, +1}
            verbose: выводить информацию о процессе обучения
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.array([-1, 1])):
            raise ValueError("Метки классов должны быть -1 и +1")
        
        if verbose:
            print(f"\nОбучение SVM с ядром '{self.kernel}' (C={self.C})")
            print(f"Размер обучающей выборки: {n_samples} объектов, {n_features} признаков")
        
        self.X_train = X
        self.y_train = y
        
        # Вычисляем матрицу Грама
        if verbose:
            print("Вычисление матрицы Грама...")
        K = compute_kernel_matrix(X, self.kernel, **self.kernel_params)
        
        # Кэшируем K * (y y^T)
        if verbose:
            print("Кэширование матрицы для оптимизации...")
        self._K_yy = K * np.outer(y, y)
        
        lambdas_init = self._initialize_lambdas(y)
        
        # Ограничения: 0 ≤ λ_i ≤ C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Ограничение: Σ λ_i y_i = 0 (используем LinearConstraint для эффективности)
        constraints = LinearConstraint(y.reshape(1, -1), lb=0, ub=0)
        
        if verbose:
            print("Решение двойственной задачи оптимизации...")
        
        # Решаем задачу оптимизации с оптимизированными параметрами
        result = minimize(
            fun=self._dual_objective,
            x0=lambdas_init,
            method='SLSQP',
            jac=self._dual_gradient,
            bounds=bounds,
            constraints=[constraints],
            options={
                'maxiter': 500,
                'disp': verbose,
                'ftol': 1e-5,
                'eps': 1.5e-8
            }
        )
        
        self._K_yy = None
        
        if not result.success:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Оптимизация не сошлась. Сообщение: {result.message}")
        
        self.lambdas = result.x
        
        # Находим опорные векторы (λ_i > epsilon)
        epsilon = 1e-5
        support_vector_indices = self.lambdas > epsilon
        self.support_vector_indices = np.where(support_vector_indices)[0]
        
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_lambdas = self.lambdas[support_vector_indices]
        
        n_support_vectors = len(self.support_vectors)
        
        if verbose:
            print(f"\nНайдено опорных векторов: {n_support_vectors} ({100*n_support_vectors/n_samples:.1f}%)")
        
        # Вычисляем смещение b
        # Используем опорные векторы, которые лежат строго внутри margin (0 < λ_i < C)
        margin_vectors_mask = (self.lambdas > epsilon) & (self.lambdas < self.C - epsilon)
        
        if np.sum(margin_vectors_mask) > 0:
            # b = y_k - Σ_i λ_i y_i K(x_i, x_k) для любого опорного вектора на границе
            margin_indices = np.where(margin_vectors_mask)[0]
            b_values = []
            
            for idx in margin_indices:
                prediction = np.sum(
                    self.lambdas * y * K[:, idx]
                )
                b_values.append(y[idx] - prediction)
            
            self.b = np.mean(b_values)
        else:
            # Если нет векторов на границе, используем все опорные векторы
            if verbose:
                print("ПРЕДУПРЕЖДЕНИЕ: Нет опорных векторов на границе margin. Используем среднее по всем опорным векторам.")
            
            support_indices = np.where(support_vector_indices)[0]
            b_values = []
            
            for idx in support_indices[:min(10, len(support_indices))]:
                prediction = np.sum(
                    self.lambdas * y * K[:, idx]
                )
                b_values.append(y[idx] - prediction)
            
            self.b = np.mean(b_values) if b_values else 0.0
        
        if verbose:
            print(f"Смещение b = {self.b:.4f}")
            print(f"Оптимизация завершена за {result.nit} итераций")
        
        return self
    
    def decision_function(self, X):
        """
        Вычисляет значение решающей функции для объектов
        
        f(x) = Σ_i λ_i y_i K(x_i, x) + b
        
        Args:
            X: тестовая выборка (n_test, d)
        
        Returns:
            Значения решающей функции (n_test,)
        """
        if self.support_vectors is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() сначала.")
        
        # Вычисляем ядро между тестовой выборкой и опорными векторами
        K_test = compute_kernel_test(self.support_vectors, X, self.kernel, **self.kernel_params)
        
        # f(x) = Σ_i λ_i y_i K(x_i, x) + b
        # K_test имеет размер (n_test, n_support_vectors)
        # Транспонируем, чтобы получить (n_support_vectors, n_test)
        decision = np.dot(self.support_vector_lambdas * self.support_vector_labels, K_test.T) + self.b
        
        return decision
    
    def predict(self, X):
        decision = self.decision_function(X)
        return np.sign(decision)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_weights(self):
        """
        Вычисляет веса w для линейного ядра
        Для нелинейных ядер возвращает None
        
        Returns:
            Вектор весов w или None
        """
        if self.kernel == 'linear':
            w = np.sum(
                self.support_vector_lambdas[:, np.newaxis] * 
                self.support_vector_labels[:, np.newaxis] * 
                self.support_vectors, 
                axis=0
            )
            return w
        else:
            return None
