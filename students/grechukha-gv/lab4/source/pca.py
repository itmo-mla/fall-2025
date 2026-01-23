import numpy as np


class PCA:
    """
    Реализация PCA (Principal Component Analysis) через сингулярное разложение (SVD).
    
    Алгоритм:
    1. Центрирование данных: X_centered = X - mean(X)
    2. Сингулярное разложение: U, S, Vt = svd(X_centered)
    3. Главные компоненты: components_ = Vt (строки - направления максимальной дисперсии)
    4. Объясненная дисперсия: explained_variance_ = S^2 / (n_samples - 1)
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_samples_ = None
        self.n_features_ = None
    
    def fit(self, X):
        X_proc = np.asarray(X)
        
        self.n_samples_, self.n_features_ = X_proc.shape
        
        # Центрирование данных
        self.mean_ = np.mean(X_proc, axis=0)
        X_centered = X_proc - self.mean_
        
        # Сингулярное разложение
        # U: (n_samples, min(n_samples, n_features))
        # S: (min(n_samples, n_features),)
        # Vt: (min(n_samples, n_features), n_features)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Сохраняем компоненты (строки Vt - направления главных компонент)
        # components_[i] - i-я главная компонента
        self.components_ = Vt
        
        self.singular_values_ = S
        
        # Вычисляем объясненную дисперсию
        # Дисперсия по каждой компоненте = S^2 / (n - 1)
        self.explained_variance_ = (S ** 2) / (self.n_samples_ - 1)
        
        # Доля объясненной дисперсии (нормализованная)
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        if self.n_components is not None:
            self.components_ = self.components_[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            self.singular_values_ = self.singular_values_[:self.n_components]
        
        self._fix_component_signs()
        
        return self
    
    def _fix_component_signs(self):
        """
        Фиксирует знак главных компонент так, чтобы элемент с максимальным
        модулем был положительным.
        """
        for i in range(self.components_.shape[0]):
            max_abs_idx = np.argmax(np.abs(self.components_[i]))
            if self.components_[i, max_abs_idx] < 0:
                self.components_[i] *= -1
    
    def transform(self, X):
        """
        Проецирует данные X в пространство главных компонент.
        
        Args:
            X: матрица признаков (n_samples, n_features)
        
        Returns:
            X_transformed: проекция (n_samples, n_components)
        """
        if self.mean_ is None:
            raise ValueError("PCA не обучен. Вызовите fit() перед transform()")
        
        X_proc = np.asarray(X)
        
        # Центрируем данные
        X_centered = X_proc - self.mean_
        
        # Проецируем на главные компоненты
        # X_transformed = X_centered @ components_.T
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Восстанавливает исходные данные из проекции.
        
        Args:
            X_transformed: проекция (n_samples, n_components)
        
        Returns:
            X_reconstructed: восстановленные данные (n_samples, n_features)
        """
        if self.mean_ is None:
            raise ValueError("PCA не обучен. Вызовите fit() перед inverse_transform()")
        
        X_transformed = np.asarray(X_transformed)
        n_components_used = X_transformed.shape[1]
        
        # Восстанавливаем: X_reconstructed = X_transformed @ components + mean
        X_reconstructed = np.dot(X_transformed, self.components_[:n_components_used]) + self.mean_
        
        return X_reconstructed
    
    def get_covariance(self):
        """
        Вычисляет ковариационную матрицу в исходном пространстве.

        # Ковариационная матрица: Σ = components.T @ diag(explained_variance) @ components
        
        Returns:
            cov: ковариационная матрица (n_features, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA не обучен. Вызовите fit() перед get_covariance()")
        
        cov = np.dot(
            self.components_.T * self.explained_variance_,
            self.components_
        )
        
        return cov


def marchenko_pastur_threshold(n_samples, n_features):
    """
    Вычисляет порог Марченко-Пастура для определения значимых компонент.
    
    Критерий Марченко-Пастура основан на теории случайных матриц и позволяет
    отделить значимые компоненты от шума в данных.
    
    Args:
        n_samples: количество объектов
        n_features: количество признаков
    
    Returns:
        lambda_plus: верхняя граница спектра для случайной матрицы
    """
    gamma = n_features / n_samples
    lambda_plus = (1 + np.sqrt(gamma)) ** 2
    return lambda_plus


def determine_effective_dimensionality(pca, variance_threshold=0.95):
    """
    Определяет эффективную размерность данных несколькими методами.
    
    Args:
        pca: обученный объект PCA
        variance_threshold: порог накопленной объясненной дисперсии (по умолчанию 95%)
    
    Returns:
        dict с информацией о размерности:
            - n_components_95: количество компонент для 95% дисперсии
            - n_components_99: количество компонент для 99% дисперсии
            - cumulative_variance: массив накопленной дисперсии
            - elbow_point: точка "локтя" (если определена)
            - marchenko_pastur: количество компонент по критерию Марченко-Пастура
    """
    if pca.explained_variance_ratio_ is None:
        raise ValueError("PCA не обучен")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    # Попытка найти точку "локтя" методом второй производной
    elbow_point = find_elbow_point(pca.explained_variance_ratio_)
    
    # Критерий Марченко-Пастура
    mp_threshold = marchenko_pastur_threshold(pca.n_samples_, pca.n_features_)
    # Нормализуем собственные значения
    normalized_eigenvalues = pca.explained_variance_ / np.sum(pca.explained_variance_) * pca.n_features_
    n_components_mp = np.sum(normalized_eigenvalues > mp_threshold)
    
    return {
        'n_components_95': n_components_95,
        'n_components_99': n_components_99,
        'cumulative_variance': cumulative_variance,
        'elbow_point': elbow_point,
        'marchenko_pastur': n_components_mp,
        'mp_threshold': mp_threshold,
    }


def find_elbow_point(variances):
    """
    Находит точку "локтя" на графике explained variance.
    
    Использует метод максимальной кривизны (second derivative).
    
    Args:
        variances: массив объясненной дисперсии по компонентам
    
    Returns:
        index: индекс точки локтя (или None, если не найдена)
    """
    if len(variances) < 3:
        return None
    
    # Вычисляем вторую производную
    first_derivative = np.diff(variances)
    second_derivative = np.diff(first_derivative)
    
    if len(second_derivative) == 0:
        return None
    
    elbow_idx = np.argmax(np.abs(second_derivative)) + 1
    
    # Проверяем, что это действительно значимая точка
    # (объясненная дисперсия >= 1%)
    if variances[elbow_idx] < 0.01:
        return None
    
    return elbow_idx


def reconstruction_error(X_original, X_reconstructed):
    """
    Вычисляет ошибку восстановления (MSE)
    """
    return np.mean((X_original - X_reconstructed) ** 2)


def compute_reconstruction_errors(X, max_components=None):
    """
    Вычисляет ошибки восстановления для разного числа компонент.
    """
    if max_components is None:
        max_components = min(X.shape)
    
    errors = []
    
    for n_comp in range(1, max_components + 1):
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        error = reconstruction_error(X, X_reconstructed)
        errors.append(error)
    
    return np.array(errors)


if __name__ == '__main__':
    print("Тест PCA на синтетических данных")
    print("-" * 50)
    
    # Генерируем данные с корреляцией
    np.random.seed(42)
    n_samples = 100
    
    X1 = np.random.randn(n_samples, 2)
    X2 = X1 + 0.5 * np.random.randn(n_samples, 2)  # коррелированные с X1
    X3 = np.random.randn(n_samples, 1) * 0.1  # малый шум
    X = np.hstack([X1, X2, X3])
    
    print(f"Размер данных: {X.shape}")
    
    pca = PCA()
    X_transformed = pca.fit_transform(X)
    
    print(f"\nОбъясненная дисперсия по компонентам:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    
    print(f"\nНакопленная дисперсия:")
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i, cum in enumerate(cumsum):
        print(f"  PC1-PC{i+1}: {cum:.4f} ({cum*100:.2f}%)")
    
    # Определяем эффективную размерность
    dim_info = determine_effective_dimensionality(pca)
    print(f"\nЭффективная размерность:")
    print(f"  95% дисперсии: {dim_info['n_components_95']} компонент")
    print(f"  99% дисперсии: {dim_info['n_components_99']} компонент")
    if dim_info['elbow_point'] is not None:
        print(f"  Точка локтя: {dim_info['elbow_point']}")
    
    # Проверяем восстановление
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    X_reconstructed = pca_2d.inverse_transform(X_2d)
    error = reconstruction_error(X, X_reconstructed)
    print(f"\nОшибка восстановления (2 компоненты): {error:.6f}")
    
    print("\nТест успешно завершен!")
