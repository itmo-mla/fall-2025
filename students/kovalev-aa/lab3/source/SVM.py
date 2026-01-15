from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, c=1, kernel_type='poly', degree=2):
        self.a = None
        self.x_ref = None
        self.y_ref = None
        self.c = c
        self.b = None
        self.kernel_type = kernel_type
        self.degree = degree
        # Храним исходные данные для визуализации
        self.x_train = None
        self.y_train = None
        self.sv_indices = None

    def kernel(self, x_i, x_j):
        if self.kernel_type == 'linear':
            return np.dot(x_i, x_j)
        elif self.kernel_type == 'poly':
            gamma = 1 / x_i.shape[0]  # как в sklearn по умолчанию
            coef0 = 1
            return (gamma * np.dot(x_i, x_j) + coef0) ** self.degree
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        N = x_train.shape[0]

        # Матрица ядра
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = self.kernel(x_train[i], x_train[j])

        def f(a):
            A = a[:, None] * a[None, :]
            Y = y_train[:, None] * y_train[None, :]
            return -np.sum(a) + 0.5 * np.sum(A * Y * K)

        constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a * y_train)}]
        bounds = [(0, self.c)] * N
        a0 = np.random.rand(N)

        result = minimize(f, a0, method='SLSQP', bounds=bounds, constraints=constraints)

        self.a = result.x
        mask = self.a > 1e-5
        self.sv_indices = np.where(mask)[0]   # индексы опорных точек
        self.a = self.a[mask]
        self.x_ref = x_train[mask]
        self.y_ref = y_train[mask]

        # вычисление b через опорные векторы
        b_values = []
        for i in range(len(self.a)):
            s = np.sum(self.a * self.y_ref * np.array([self.kernel(self.x_ref[j], self.x_ref[i]) for j in range(len(self.a))]))
            b_values.append(self.y_ref[i] - s)
        self.b = np.mean(b_values)

    def activation(self, y):
        return np.sign(y)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            k = np.array([self.kernel(x_sv, X[i]) for x_sv in self.x_ref])
            y_pred[i] = np.sum(self.a * self.y_ref * k) + self.b
        return self.activation(y_pred)

 