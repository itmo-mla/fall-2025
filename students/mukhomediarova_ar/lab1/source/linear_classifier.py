from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


Array = np.ndarray


def add_bias_column(x: Array) -> Array:
    """
    Добавляет столбец единиц к матрице признаков.
    """
    bias = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([bias, x])


def margin_single(w: Array, x_i: Array, y_i: float) -> float:
    """
    Отступ одного объекта: M_i = y_i * <w, x_i>.

    Предполагается, что x_i уже содержит bias-координату.
    """
    return float(y_i * np.dot(w, x_i))


def margins(w: Array, x: Array, y: Array) -> Array:
    """
    Отступы всех объектов выборки.
    """
    return y * (x @ w)


def quadratic_margin_loss(margin_value: float) -> float:
    """
    Квадратичная функция потерь по отступу:
    L(M) = max(0, 1 - M)^2.
    """
    diff = 1.0 - margin_value
    if diff <= 0.0:
        return 0.0
    return float(diff * diff)


def quadratic_margin_loss_vectorized(margin_values: Array) -> float:
    """
    Средняя квадратичная потеря по отступам для всей выборки.
    """
    diff = 1.0 - margin_values
    diff = np.where(diff > 0.0, diff, 0.0)
    return float(np.mean(diff * diff))


def l2_penalty(w: Array, alpha: float) -> float:
    """
    L2‑регуляризация без штрафа на bias‑координату.
    """
    if alpha <= 0.0:
        return 0.0
    # Не штрафуем bias (w[0])
    return float(0.5 * alpha * np.dot(w[1:], w[1:]))


def l2_grad(w: Array, alpha: float) -> Array:
    if alpha <= 0.0:
        return np.zeros_like(w)
    g = alpha * w
    g[0] = 0.0
    return g


def full_loss_and_grad(
    w: Array,
    x: Array,
    y: Array,
    alpha: float = 0.0,
) -> Tuple[float, Array]:
    """
    Эмпирический риск (средняя квадратичная потеря по отступу) + L2‑регуляризация
    и его градиент по w.
    """
    m = margins(w, x, y)
    diff = 1.0 - m
    mask = diff > 0.0

    # Потери
    loss_margin = np.mean((diff[mask]) ** 2) if np.any(mask) else 0.0
    loss_reg = l2_penalty(w, alpha)
    loss = loss_margin + loss_reg

    # Градиент по отступу
    grad = np.zeros_like(w)
    if np.any(mask):
        # d/dw L_i = -2 * (1 - M_i) * y_i * x_i
        scaled = -2.0 * diff[mask] * y[mask]
        grad = (scaled[:, None] * x[mask]).mean(axis=0)

    grad += l2_grad(w, alpha)
    return float(loss), grad


def _single_sample_grad(
    w: Array,
    x_i: Array,
    y_i: float,
    alpha: float = 0.0,
) -> Array:
    """
    Градиент по одному объекту (без усреднения по батчу).
    """
    m = margin_single(w, x_i, y_i)
    diff = 1.0 - m
    if diff <= 0.0:
        grad = np.zeros_like(w)
    else:
        grad = -2.0 * diff * y_i * x_i
    grad += l2_grad(w, alpha)
    return grad


def initialize_weights_random(n_features_with_bias: int, scale: float = 0.01) -> Array:
    """
    Случайная инициализация весов.
    """
    rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=scale, size=n_features_with_bias)


def initialize_weights_correlation(x: Array, y: Array, scale: float = 0.1) -> Array:
    """
    Инициализация весов по корреляции признаков с целевой переменной.

    Предполагается, что `x` не содержит bias‑координаты.
    """
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean()

    num = (x_centered * y_centered[:, None]).sum(axis=0)
    denom = np.linalg.norm(x_centered, axis=0) * float(np.linalg.norm(y_centered))
    denom[denom == 0.0] = 1.0

    corr = num / denom
    # bias отдельно
    w_no_bias = scale * corr
    w = np.empty(w_no_bias.shape[0] + 1, dtype=float)
    w[0] = 0.0
    w[1:] = w_no_bias
    return w


@dataclass
class SGDConfig:
    learning_rate: float = 0.01
    n_epochs: int = 100
    batch_size: int = 1
    alpha_l2: float = 0.0
    momentum: float = 0.0
    use_margin_sampling: bool = False
    ema_alpha: Optional[float] = None
    random_state: Optional[int] = None


def sgd_train(
    x: Array,
    y: Array,
    w_init: Array,
    config: SGDConfig,
) -> Tuple[Array, List[float], Optional[float]]:
    """
    Стохастический градиентный спуск с опциональной инерцией (momentum),
    L2‑регуляризацией и рекуррентной оценкой функционала качества (EMA).

    Если use_margin_sampling=True, объекты предъявляются в порядке усложнённости
    по модулю отступа (чаще выбираются объекты с малым |M_i|).
    """
    rng = np.random.default_rng(config.random_state)
    w = w_init.copy()
    v = np.zeros_like(w)

    n_samples = x.shape[0]
    history: List[float] = []
    ema_value: Optional[float] = None

    for epoch in range(config.n_epochs):
        if config.use_margin_sampling:
            m = margins(w, x, y)
            probs = 1.0 / (np.abs(m) + 1e-6)
            probs /= probs.sum()
            indices = rng.choice(n_samples, size=n_samples, replace=False, p=probs)
        else:
            indices = rng.permutation(n_samples)

        total_epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            batch_idx = indices[start : start + config.batch_size]
            x_b, y_b = x[batch_idx], y[batch_idx]

            grad_batch = np.zeros_like(w)
            batch_loss = 0.0
            for x_i, y_i in zip(x_b, y_b):
                m_i = margin_single(w, x_i, y_i)
                batch_loss += quadratic_margin_loss(m_i)
                grad_batch += _single_sample_grad(
                    w,
                    x_i,
                    float(y_i),
                    alpha=config.alpha_l2,
                )

            size_b = len(x_b)
            if size_b > 0:
                grad_batch /= size_b
                batch_loss /= size_b

            # Momentum
            if config.momentum > 0.0:
                v = config.momentum * v + grad_batch
                step = v
            else:
                step = grad_batch

            w = w - config.learning_rate * step

            total_epoch_loss += batch_loss
            n_batches += 1

            if config.ema_alpha is not None:
                if ema_value is None:
                    ema_value = batch_loss
                else:
                    ema_value = (
                        config.ema_alpha * batch_loss
                        + (1.0 - config.ema_alpha) * ema_value
                    )

        mean_epoch_loss = total_epoch_loss / max(n_batches, 1)
        history.append(float(mean_epoch_loss))

    return w, history, ema_value


def multi_start_sgd(
    x: Array,
    y: Array,
    n_starts: int,
    base_config: SGDConfig,
) -> Tuple[Array, float]:
    """
    Мультистарт: несколько запусков SGD со случайной инициализацией.
    Возвращает лучшие веса и ошибку на train.
    """
    best_w: Optional[Array] = None
    best_error: float = 1.0

    for _ in range(n_starts):
        w0 = initialize_weights_random(x.shape[1])
        w_trained, _, _ = sgd_train(x, y, w0, base_config)
        m_train = margins(w_trained, x, y)
        error = float(np.mean(m_train < 0.0))
        if error < best_error or best_w is None:
            best_error = error
            best_w = w_trained

    assert best_w is not None
    return best_w, best_error


def steepest_gradient_descent(
    x: Array,
    y: Array,
    w_init: Array,
    alpha_l2: float = 0.0,
    n_epochs: int = 50,
    initial_step: float = 1.0,
    armijo_c: float = 1e-4,
    backtrack_factor: float = 0.5,
) -> Tuple[Array, List[float]]:
    """
    Скорейший градиентный спуск с одномерным поиском шага вдоль антиградиента
    (backtracking line search по условию Армихо).
    """
    w = w_init.copy()
    history: List[float] = []

    for _ in range(n_epochs):
        loss, grad = full_loss_and_grad(w, x, y, alpha=alpha_l2)
        direction = -grad
        grad_norm_sq = float(np.dot(grad, grad))
        if grad_norm_sq == 0.0:
            history.append(loss)
            break

        step = initial_step
        while True:
            new_w = w + step * direction
            new_loss, _ = full_loss_and_grad(new_w, x, y, alpha=alpha_l2)
            if new_loss <= loss - armijo_c * step * grad_norm_sq or step < 1e-8:
                w = new_w
                loss = new_loss
                break
            step *= backtrack_factor

        history.append(loss)

    return w, history


def predict(w: Array, x: Array) -> Array:
    """
    Предсказания меток {-1, 1}.
    """
    scores = x @ w
    y_pred = np.where(scores >= 0.0, 1.0, -1.0)
    return y_pred


def classification_error(y_true: Array, y_pred: Array) -> float:
    """
    Ошибка классификации (доля неправильных ответов).
    """
    assert y_true.shape == y_pred.shape
    return float(np.mean(y_true != y_pred))


def accuracy_score(y_true: Array, y_pred: Array) -> float:
    return 1.0 - classification_error(y_true, y_pred)


def confusion_matrix(y_true: Array, y_pred: Array) -> Dict[str, float]:
    """
    Confusion matrix и базовые метрики для классов {-1, 1}.
    """
    tp = float(np.sum((y_true == 1.0) & (y_pred == 1.0)))
    tn = float(np.sum((y_true == -1.0) & (y_pred == -1.0)))
    fp = float(np.sum((y_true == -1.0) & (y_pred == 1.0)))
    fn = float(np.sum((y_true == 1.0) & (y_pred == -1.0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def baseline_least_squares(x: Array, y: Array, alpha_l2: float = 0.0) -> Array:
    """
    Простая эталонная линейная модель:
    минимизация ||y - Xw||^2 + alpha * ||w||^2 (кроме bias).
    """
    # Регуляризация только для весов (кроме bias).
    if alpha_l2 <= 0.0:
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        return w

    n_features = x.shape[1]
    reg = alpha_l2 * np.eye(n_features)
    reg[0, 0] = 0.0  # не штрафуем bias
    a = x.T @ x + reg
    b = x.T @ y
    return np.linalg.solve(a, b)

