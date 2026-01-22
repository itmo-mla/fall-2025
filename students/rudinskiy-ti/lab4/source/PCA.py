import numpy as np
import pandas as pd

def find_elbow(y):
    """
    Поиск локтя через геометрические соображения
    
    :param y: Список значений на графике
    """
    y = np.array(y)
    x = np.arange(len(y))

    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_len = np.hypot(line_vec[0], line_vec[1])
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    distances = []
    for i in range(len(x)):
        point_vec = np.array([x[i] - x[0], y[i] - y[0]])
        proj_len = np.dot(point_vec, line_unit)
        proj = proj_len * line_unit
        dist = np.hypot(point_vec[0] - proj[0], point_vec[1] - proj[1])
        distances.append(dist)
    return int(np.argmax(distances))

class HandmadePCA:

  def __init__(self, optimal_value=True, n_components=5):
    """
    Инициализация модели
    
    :param optimal_value: Флаг для поиска оптимального значения колчества компонент
    :param n_components: Количество компонент
    """
    self.n_components = n_components
    self.optimal_value = optimal_value

  def fit(self, x):
    X_np = x.to_numpy()
    N = len(X_np)
    self.mean_ = X_np.mean(axis=0)
    X_centered = X_np - self.mean_
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    D = np.diag(s)
    Em = []
    if self.optimal_value:
      Em = self.optimal_m(s)
      m = find_elbow(Em)
      self.n_components = m
    else:
      m = self.n_components
    G = (U @ D)[:, :m]
    Vt = Vt[:m, :]
    self.G = G
    self.Vt = Vt
    self.explained_variance = (s**2 / (N - 1))[:m]
    self.explained_variance_ratio = (s**2 / np.sum(s**2))[:m]
    self.U = U
    return Em

  def transform(self, X):
    X_np = X.to_numpy()
    return (X_np - self.mean_) @ self.Vt.T

  def optimal_m(self, s):
    s_sq = s**2
    total = s_sq.sum()
    Em = []
    for i in range(len(s_sq)):
        residual = s_sq[i+1:].sum()  # после i-й компоненты (0-based)
        Em.append(residual / total)
    return Em
