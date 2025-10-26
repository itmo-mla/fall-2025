# реализация KNN с Parzen (переменная ширина)
"""KNN с методом окна Парзена переменной ширины (гауссово ядро).

Интерфейс похож на sklearn: класс KNNParzen с методами fit(X, y) и predict(X, k).
"""
import numpy as np
from collections import defaultdict


class KNNParzen:
    def __init__(self, metric=None, eps=1e-12):
        # metric: функция расстояния dist(X1, X2) -> (n1, n2) матрица или None для Евклида
        self.metric = metric
        self.eps = eps

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def _pairwise_distances(self, A, B):
        if self.metric is None:
            # Евклидово расстояние (быстрое векторное)
            A2 = np.sum(A * A, axis=1)[:, None]
            B2 = np.sum(B * B, axis=1)[None, :]
            D2 = A2 + B2 - 2 * (A @ B.T)
            D2 = np.maximum(D2, 0.0)
            return np.sqrt(D2)
        else:
            # пользовательская метрика
            return self.metric(A, B)

    def predict(self, X, k=5):
        X = np.asarray(X)
        n_train = self.X.shape[0]
        if k < 1:
            raise ValueError("k must be >= 1")
        D = self._pairwise_distances(X, self.X)  # shape (n_test, n_train)
        # Для каждого теста найдем (k+1)-ю дистанцию (по условию переменной ширины используется d_{k+1})
        sorted_idx = np.argsort(D, axis=1)
        sorted_d = np.take_along_axis(D, sorted_idx, axis=1)
        # d_{k+1}: если k >= n_train -> берем max расстояние
        kplus1 = np.minimum(k, n_train - 1)  # индекс k (0-based) для d_{k+1}
        d_kplus1 = sorted_d[:, kplus1] + self.eps  # (n_test,)

        preds = []
        for i in range(X.shape[0]):
            di = D[i]
            h = d_kplus1[i]
            # веса: гауссово ядро K(r) = exp(-2 * r^2), где r = dist/h
            r = di / h
            w = np.exp(-2.0 * (r ** 2))
            # суммируем веса по классам
            votes = defaultdict(float)
            for idx_tr, wi in enumerate(w):
                votes[self.y[idx_tr]] += wi
            # argmax голосов
            # в случае равенства — выбрать минимальный класс-лейбл для детерминированности
            best = max(sorted(votes.items(), key=lambda kv: (kv[1], -float(kv[0]))), key=lambda kv: kv[1])[0]
            preds.append(best)
        return np.array(preds)

    def predict_single_loocv(self, idx, k=5):
        # классифицировать train[idx] используя X\{idx}
        X_query = self.X[idx:idx+1]
        X_train = np.delete(self.X, idx, axis=0)
        y_train = np.delete(self.y, idx, axis=0)
        # временно используем новый объект класса для простоты
        tmp = KNNParzen(metric=self.metric, eps=self.eps)
        tmp.fit(X_train, y_train)
        return tmp.predict(X_query, k=k)[0]
