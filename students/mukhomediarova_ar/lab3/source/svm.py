from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - scipy может быть не установлен в среде проверки
    from scipy.optimize import minimize
except Exception:  # type: ignore
    minimize = None  # type: ignore


Array = np.ndarray


def linear_kernel(x: Array, z: Array) -> Array:
    """Линейное ядро K(x, z) = <x, z>."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return x @ z.T


def make_rbf_kernel(gamma: float = 1.0) -> Callable[[Array, Array], Array]:
    """Создаёт RBF‑ядро K(x, z) = exp(-gamma * ||x - z||^2)."""

    def rbf(x: Array, z: Array) -> Array:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        x_sq = np.sum(x * x, axis=1)[:, None]
        z_sq = np.sum(z * z, axis=1)[None, :]
        dist_sq = x_sq + z_sq - 2.0 * (x @ z.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    return rbf


def accuracy_score(y_true: Array, y_pred: Array) -> float:
    """Простая accuracy для меток {-1, 1} или {0, 1}."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    return float(np.mean(y_true == y_pred))


@dataclass
class KernelSVM:
    """SVM в двойственной постановке с поддержкой произвольного ядра.

    Решаем задачу (soft‑margin):

        max_α  W(α) = Σ_i α_i - 1/2 Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
        при  0 <= α_i <= C,  Σ_i α_i y_i = 0.

    Мы минимизируем -W(α) с помощью scipy.optimize.minimize (SLSQP).
    """

    c: float = 1.0
    kernel: Callable[[Array, Array], Array] = linear_kernel
    tol: float = 1e-5
    max_iter: int = 200

    # Поля, заполняемые после обучения
    alphas_: Optional[Array] = None
    support_indices_: Optional[Array] = None
    support_vectors_: Optional[Array] = None
    support_labels_: Optional[Array] = None
    b_: Optional[float] = None

    def _solve_dual(self, k_mat: Array, y: Array) -> Array:
        if minimize is None:
            raise RuntimeError(
                "scipy.optimize.minimize не доступен. "
                "Для этой реализации SVM необходим scipy."
            )

        n = y.shape[0]
        y = y.astype(float)

        def objective(alpha: Array) -> float:
            # 0.5 * (α * y)^T K (α * y) - Σ α_i
            ay = alpha * y
            return 0.5 * float(ay @ (k_mat @ ay)) - float(alpha.sum())

        def grad(alpha: Array) -> Array:
            ay = alpha * y
            # ∂/∂α_i: Σ_j α_i α_j y_i y_j K_ij -> (K (α * y))_i * y_i
            k_ay = k_mat @ ay
            return y * k_ay - 1.0

        # Ограничения: 0 <= α_i <= C
        bounds = [(0.0, self.c) for _ in range(n)]

        # Линейное ограничение: Σ α_i y_i = 0
        constraints = {
            "type": "eq",
            "fun": lambda a: float(a @ y),
            "jac": lambda a: y,
        }

        alpha0 = np.zeros(n, dtype=float)

        res = minimize(
            objective,
            alpha0,
            jac=grad,
            bounds=bounds,
            constraints=[constraints],
            method="SLSQP",
            options={"maxiter": self.max_iter, "ftol": 1e-9, "disp": False},
        )

        if not res.success:
            print("Предупреждение: оптимизатор SLSQP завершился с сообщением:", res.message)

        alpha_opt = np.asarray(res.x, dtype=float)
        # Обнуляем очень маленькие значения
        alpha_opt[alpha_opt < self.tol] = 0.0
        return alpha_opt

    def fit(self, x: Array, y: Array) -> "KernelSVM":
        """Обучение SVM по выборке (x, y), где y ∈ {-1, 1}."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape[0] != y.shape[0]:
            raise ValueError("Количество объектов и меток не совпадает")

        # Матрица ядра K_ij = K(x_i, x_j)
        k_mat = self.kernel(x, x)

        alphas = self._solve_dual(k_mat, y)
        support_mask = alphas > self.tol
        support_indices = np.nonzero(support_mask)[0]

        if support_indices.size == 0:
            raise RuntimeError("Не удалось найти ни одного опорного вектора")

        self.alphas_ = alphas
        self.support_indices_ = support_indices
        self.support_vectors_ = x[support_indices]
        self.support_labels_ = y[support_indices]

        # Восстановление смещения b.
        # Используем только те опорные векторы, для которых 0 < α_i < C.
        margin_mask = (alphas > self.tol) & (alphas < self.c - self.tol)
        margin_indices = np.nonzero(margin_mask)[0]

        if margin_indices.size == 0:
            # Если все опорные векторы на границе, используем все опорные.
            margin_indices = support_indices

        b_values = []
        for i in margin_indices:
            # y_i - Σ_j α_j y_j K(x_j, x_i)
            k_row = k_mat[i, :]
            decision_without_b = np.sum(alphas * y * k_row)
            b_values.append(float(y[i] - decision_without_b))

        self.b_ = float(np.mean(b_values))
        return self

    # Вспомогательные методы -------------------------------------------------

    def _decision_function_raw(self, x: Array) -> Array:
        """Линейная комбинация ядровых функций без сигнала об ошибках."""
        if self.support_vectors_ is None or self.support_labels_ is None or self.alphas_ is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        x = np.asarray(x, dtype=float)
        sv = self.support_vectors_
        y_sv = self.support_labels_
        alphas = self.alphas_[self.support_indices_]  # type: ignore[index]

        k = self.kernel(x, sv)  # shape (n_samples, n_support)
        # Σ_j α_j y_j K(x, x_j)
        scores = k @ (alphas * y_sv)
        return scores

    def decision_function(self, x: Array) -> Array:
        """Значение решающей функции f(x) = Σ_j α_j y_j K(x_j, x) + b."""
        scores = self._decision_function_raw(x)
        if self.b_ is None:
            raise RuntimeError("b_ не инициализирован. Вызовите fit() перед predict().")
        return scores + self.b_

    def predict(self, x: Array) -> Array:
        """Предсказания меток {-1, 1} по знаку решающей функции."""
        f = self.decision_function(x)
        return np.where(f >= 0.0, 1.0, -1.0)

    def margin(self, x: Array, y: Array) -> Array:
        """Отступы M_i = y_i * f(x_i)."""
        f = self.decision_function(x)
        y = np.asarray(y, dtype=float)
        return y * f

