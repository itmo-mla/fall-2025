import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def gaussian_kernel(distance, h):
    """
    Гауссово ядро Парзена.
    
    Args:
        distance: расстояние до объекта
        h: ширина окна (bandwidth)
    
    Returns:
        float: значение ядра
    """
    if h == 0:
        return 0.0
    return np.exp(-(distance ** 2) / (2 * h ** 2))


def rectangular_kernel(distance, h):
    """
    Прямоугольное (uniform) ядро.
    
    Args:
        distance: расстояние до объекта
        h: ширина окна
    
    Returns:
        float: 1.0 если distance <= h, иначе 0.0
    """
    if h == 0:
        return 0.0
    return 1.0 if distance <= h else 0.0


def triangular_kernel(distance, h):
    """
    Треугольное ядро.
    
    Args:
        distance: расстояние до объекта
        h: ширина окна
    
    Returns:
        float: значение ядра
    """
    if h == 0:
        return 0.0
    normalized = distance / h
    return max(0, 1 - normalized)


def epanechnikov_kernel(distance, h):
    """
    Ядро Епанечникова (квадратичное).
    
    Args:
        distance: расстояние до объекта
        h: ширина окна
    
    Returns:
        float: значение ядра
    """
    if h == 0:
        return 0.0
    normalized = distance / h
    return max(0, 0.75 * (1 - normalized ** 2))


def quartic_kernel(distance, h):
    """
    Квартическое (биквадратное) ядро.
    
    Args:
        distance: расстояние до объекта
        h: ширина окна
    
    Returns:
        float: значение ядра
    """
    if h == 0:
        return 0.0
    normalized = distance / h
    return max(0, (15/16) * (1 - normalized ** 2) ** 2)


KERNELS = {
    'gaussian': gaussian_kernel,
    'rectangular': rectangular_kernel,
    'triangular': triangular_kernel,
    'epanechnikov': epanechnikov_kernel,
    'quartic': quartic_kernel,
}


class KNNParzenWindow:
    """
    KNN классификатор с методом окна Парзена переменной ширины.
    
    Использует различные ядра для взвешенного голосования соседей.
    Ширина окна h определяется как расстояние до k-го ближайшего соседа.
    """
    
    def __init__(self, k=5, kernel='gaussian', class_weights='balanced'):
        """
        Args:
            k: количество ближайших соседей
            kernel: тип ядра ('gaussian', 'rectangular', 'triangular', 'epanechnikov', 'quartic')
            class_weights: 'balanced' для автоматической балансировки, dict для ручных весов или None для отсутствия весов
        """
        self.k = k
        self.kernel_type = kernel
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel: {kernel}. Available: {list(KERNELS.keys())}")
        self.kernel_func = KERNELS[kernel]
        self.X_train = None
        self.y_train = None
        self.class_weights_param = class_weights
        self.class_weights_ = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        
        # Вычисляем веса классов
        if self.class_weights_param == 'balanced':
            class_counts = np.bincount(self.y_train)
            total = len(self.y_train)
            self.class_weights_ = {
                cls: total / (len(self.classes_) * count) 
                for cls, count in enumerate(class_counts)
            }
        elif isinstance(self.class_weights_param, dict):
            self.class_weights_ = self.class_weights_param
        else:
            self.class_weights_ = {cls: 1.0 for cls in self.classes_}
        
        return self
    
    def predict_single(self, x):
        # Предсказывает класс для одного объекта.
        # Вычисляем расстояния до всех объектов обучающей выборки.
        distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
        
        # Находим индексы k ближайших соседей
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # Ширина окна h = расстояние до k-го соседа
        h = k_nearest_distances[-1] if len(k_nearest_distances) > 0 else 1.0
        
        # Если h = 0 (все соседи в одной точке), используем простое голосование
        if h == 0:
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            return unique[np.argmax(counts)]
        
        # Вычисляем веса с помощью выбранного ядра
        weights = np.array([self.kernel_func(d, h) for d in k_nearest_distances])
        
        # Взвешенное голосование по классам с учетом весов классов
        class_votes = {}
        for label, weight in zip(k_nearest_labels, weights):
            class_weight = self.class_weights_.get(label, 1.0)
            class_votes[label] = class_votes.get(label, 0) + weight * class_weight
        
        # Возвращаем класс с максимальным весом
        return max(class_votes, key=class_votes.get)
    
    def predict(self, X):
        """
        Предсказывает классы для массива объектов.
        
        Args:
            X: матрица признаков
        
        Returns:
            numpy.ndarray: массив предсказанных классов
        """
        return np.array([self.predict_single(x) for x in X])
    
    def score(self, X, y):
        """
        Вычисляет accuracy на выборке.
        
        Args:
            X: матрица признаков
            y: истинные метки классов
        
        Returns:
            float: accuracy (доля правильных предсказаний)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class KNNParzenWindowEfficient:
    """
    Эффективная реализация KNN с окном Парзена (векторизованная).
    Использует матричные операции для ускорения вычислений.
    Поддерживает взвешенное голосование для балансировки классов.
    """
    
    def __init__(self, k=5, kernel='gaussian', class_weights='balanced'):
        """
        Args:
            k: количество ближайших соседей
            kernel: тип ядра
            class_weights: 'balanced' для автоматической балансировки, dict для ручных весов или None для отсутствия весов
        """
        self.k = k
        self.kernel_type = kernel
        if kernel not in KERNELS:
            raise ValueError(f"Unknown kernel: {kernel}. Available: {list(KERNELS.keys())}")
        self.kernel_func = KERNELS[kernel]
        self.X_train = None
        self.y_train = None
        self.class_weights_param = class_weights
        self.class_weights_ = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        
        # Вычисляем веса классов
        if self.class_weights_param == 'balanced':
            # Веса обратно пропорциональны частоте класса
            class_counts = np.bincount(self.y_train)
            total = len(self.y_train)
            self.class_weights_ = {
                cls: total / (len(self.classes_) * count) 
                for cls, count in enumerate(class_counts)
            }
            print(f"Веса классов (balanced): {self.class_weights_}")
        elif isinstance(self.class_weights_param, dict):
            self.class_weights_ = self.class_weights_param
        else:
            self.class_weights_ = {cls: 1.0 for cls in self.classes_}
        
        return self
    
    def compute_distances(self, X):
        """
        Вычисляет матрицу расстояний между X и обучающей выборкой.
        
        Args:
            X: матрица объектов для предсказания (n_samples, n_features)
        
        Returns:
            numpy.ndarray: матрица расстояний (n_samples, n_train)
        """
        # Векторизованное вычисление евклидовых расстояний
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        X_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
        train_norm_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        cross_term = np.dot(X, self.X_train.T)
        
        distances_sq = X_norm_sq + train_norm_sq - 2 * cross_term
        distances_sq = np.maximum(distances_sq, 0)  # Устраняем возможные отрицательные значения из-за погрешностей
        
        return np.sqrt(distances_sq)
    
    def predict(self, X):
        """
        Предсказывает классы для массива объектов (векторизованная версия).
        
        Args:
            X: матрица признаков (n_samples, n_features)
        
        Returns:
            numpy.ndarray: массив предсказанных классов
        """
        X = np.array(X)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.y_train.dtype)
        
        # Вычисляем все расстояния за один раз
        distances = self.compute_distances(X)
        
        # Для каждого объекта находим k ближайших соседей
        for i in range(n_samples):
            k_nearest_indices = np.argsort(distances[i])[:self.k]
            k_nearest_distances = distances[i, k_nearest_indices]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Ширина окна
            h = k_nearest_distances[-1] if len(k_nearest_distances) > 0 else 1.0
            
            if h == 0:
                unique, counts = np.unique(k_nearest_labels, return_counts=True)
                predictions[i] = unique[np.argmax(counts)]
                continue
            
            # Вычисляем веса с помощью выбранного ядра
            weights = np.array([self.kernel_func(d, h) for d in k_nearest_distances])
            
            # Взвешенное голосование с учетом весов классов
            class_votes = {}
            for label, weight in zip(k_nearest_labels, weights):
                # Умножаем вес расстояния на вес класса
                class_weight = self.class_weights_.get(label, 1.0)
                class_votes[label] = class_votes.get(label, 0) + weight * class_weight
            
            predictions[i] = max(class_votes, key=class_votes.get)
        
        return predictions
    
    def score(self, X, y):
        """
        Вычисляет accuracy на выборке.
        
        Args:
            X: матрица признаков
            y: истинные метки классов
        
        Returns:
            float: accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

