from .utils import euclidean_distance, gaussian_kernel, triangular_kernel, epanechnikov_kernel, KERNELS


class KNNParzen:
    def __init__(self, kernel='gaussian'):
        """
        Инициализация KNN с методом Парзена

        """
        self.X_train = None
        self.y_train = None
        self.kernel = KERNELS[kernel]
        self.class_counts = None

    def fit(self, X, y):
        """Обучение модели"""
        self.X_train = X
        self.y_train = y

    def predict_variable_window(self, X_test, k):
        """Метод Парзена с переменной шириной окна (используются только k ближайших соседей)"""
        predictions = []
        for i, x in enumerate(X_test):
            # Вычисляем расстояния до всех обучающих объектов
            distances = [(euclidean_distance(x, x_train), self.y_train[j])
                            for j, x_train in enumerate(self.X_train)]

            # Сортируем по расстоянию
            distances.sort(key=lambda d: d[0])

            # Берем только k ближайших соседей
            k_nearest = distances[:k]

            # Ширина окна = расстояние до k-го соседа
            h_k = k_nearest[-1][0] if k_nearest else distances[0][0]

            # Если расстояние до k-го соседа нулевое, используем небольшое значение
            if h_k == 0:
                h_k = 1e-10

            weights = {}
            for dist, label in k_nearest:
                r = dist / h_k
                weight = self.kernel(r)
                # Для точек на границе (r=1) с компактными ядрами добавляем минимальный вес
                if weight == 0:
                    weight = 1e-10                    
                weights[label] = weights.get(label, 0) + weight

            # Выбираем класс с максимальным весом
            if weights:
                predicted = max(weights.items(), key=lambda w: w[1])[0]
            else:
                # Если веса не определены (например, k=0), берем ближайшего соседа
                predicted = distances[0][1]
            
            predictions.append(predicted)

        return predictions
