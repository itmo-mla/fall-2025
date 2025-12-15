# lab1_linear_sgd_step_by_step.py
# Один файл. Задания выполняются ПО ПОРЯДКУ, каждый шаг — отдельная функция train_step_X()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # папка, где лежит lab1.py
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# DATASET
def load_data(path="lab1/data.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    y = df["diagnosis"].map({"M": 1, "B": -1}).values
    X = df.drop(columns=["diagnosis"]).values.astype(float)

    return X, y

def standardize_train_test(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0) + 1e-12
    return (Xtr - mu) / sigma, (Xte - mu) / sigma

def add_bias(X):
    # bias-feature = -1 (как в лекции)
    return np.hstack([-np.ones((X.shape[0], 1)), X])


# margin, loss, grad

def margins(X, y, w):
    # Задание 2: M_i = y_i * <x_i, w>
    return y * (X @ w)

def hinge_loss(m):
    # hinge: max(0, 1 - M)
    return np.maximum(0.0, 1.0 - m)

def loss_value(X, y, w):
    # обычный (не-EMA) лосс по всей выборке
    m = margins(X, y, w)
    return hinge_loss(m).mean()

def grad_hinge_one(x_i, y_i, w):
    # Задание 3: градиент hinge по одному объекту
    # L = max(0, 1 - y<w,x>)
    # если 1 - y<w,x> > 0 => grad = -y*x
    # иначе 0
    margin = y_i * (x_i @ w)
    if margin < 1.0:
        return -y_i * x_i
    return np.zeros_like(w)


# =============================================================================
# 2) Визуализация отступов (Задание 2)
# =============================================================================

def plot_sorted_margins(X, y, w, title, filename):
    M = margins(X, y, w)
    M_sorted = np.sort(M)

    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(M_sorted)), M_sorted, s=10, alpha=0.7)
    plt.axhline(0.0, linewidth=2)
    plt.axhline(1.0, linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Object rank")
    plt.ylabel("Margin")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()


# БАЗОВЫЙ SGD
def sgd_basic(X, y, lr=1e-2, epochs=5, seed=42):
    """
    Задание 9 (частично): обучить линейный классификатор.
    Это САМЫЙ базовый SGD:
      - случайный объект
      - шаг w := w - lr * grad
      - сохраняем обычный loss по всей выборке (для графика)
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])

    loss_hist = []

    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)
            g = grad_hinge_one(X[i], y[i], w)
            w -= lr * g

        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist


# Рекуррентная оценка функционала качества (EMA)

def sgd_with_ema(X, y, lr=1e-2, epochs=5, ema_lambda=0.05, seed=42):
    """
    Добавляем Задание 4: EMA(Q)
    Q := lambda * xi + (1-lambda)*Q, где xi - loss на текущем объекте (или батче).
    Параллельно оставляем обычный loss по эпохам.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])

    loss_hist = []      # обычный loss по эпохам
    q_ema_hist = []     # EMA по шагам (можно прореживать для графика)

    # инициализация Q по случайным объектам
    idx0 = rng.choice(len(X), size=min(32, len(X)), replace=False)
    q = hinge_loss(margins(X[idx0], y[idx0], w)).mean()
    q_ema_hist.append(q)

    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)

            # шаг SGD
            g = grad_hinge_one(X[i], y[i], w)
            w -= lr * g

            # xi = loss на текущем объекте
            mi = y[i] * (X[i] @ w)
            xi = max(0.0, 1.0 - mi)
            q = ema_lambda * xi + (1.0 - ema_lambda) * q
            q_ema_hist.append(q)

        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist, q_ema_hist


# SGD с инерцией (Momentum)
def sgd_with_momentum(X, y, lr=1e-2, epochs=5, gamma=0.9, seed=42):
    """
    v := gamma*v + (1-gamma)*grad
    w := w - lr*v
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])
    v = np.zeros_like(w)

    loss_hist = []
    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)
            g = grad_hinge_one(X[i], y[i], w)
            v = gamma * v + (1.0 - gamma) * g
            w -= lr * v

        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist


# L2 регуляризация
def grad_hinge_one_l2(x_i, y_i, w, tau):
    g = grad_hinge_one(x_i, y_i, w)

    # + tau*w (обычно bias не регуляризуют)
    reg = tau * w.copy()
    reg[0] = 0.0
    return g + reg

def sgd_with_l2(X, y, lr=1e-2, epochs=5, tau=1e-2, seed=42):
    """
    Добавляем L2:
      L_total = hinge + (tau/2)||w||^2
      grad += tau*w
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])

    loss_hist = []
    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)
            g = grad_hinge_one_l2(X[i], y[i], w, tau)
            w -= lr * g

        # считаем полный loss (hinge + l2)
        base = loss_value(X, y, w)
        l2 = 0.5 * tau * np.sum(w[1:] ** 2)
        loss_hist.append(base + l2)

    return w, loss_hist


# Скорейший градиентный спуск
def sgd_fastest(X, y, epochs=5, seed=42):
    """
    Для одного объекта: h* = 1 / ||x||^2
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])

    loss_hist = []
    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)
            x_i, y_i = X[i], y[i]
            g = grad_hinge_one(x_i, y_i, w)

            h = 1.0 / (np.dot(x_i, x_i) + 1e-12)
            w -= h * g

        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist


# Предъявление по модулю отступа
def choose_index_by_margin_abs(X, y, w, rng):
    M_abs = np.abs(margins(X, y, w))
    probs = 1.0 / (M_abs + 1e-6)
    probs /= probs.sum()
    return rng.choice(len(X), p=probs)

def sgd_margin_sampling(X, y, lr=1e-2, epochs=5, seed=42):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])

    loss_hist = []
    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = choose_index_by_margin_abs(X, y, w, rng)
            g = grad_hinge_one(X[i], y[i], w)
            w -= lr * g

        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist


# Инициализация через корреляцию
def init_weights_corr(X, y, eps=1e-12):
    # w_j := <y, f_j> / <f_j, f_j>
    w = np.zeros(X.shape[1])
    for j in range(1, X.shape[1]):
        fj = X[:, j]
        w[j] = (y @ fj) / (fj @ fj + eps)
    return w

def sgd_with_corr_init(X, y, lr=1e-2, epochs=5, seed=42):
    rng = np.random.default_rng(seed)
    w = init_weights_corr(X, y)

    loss_hist = []
    n = len(X)
    for _ in range(epochs):
        for _ in range(n):
            i = rng.integers(0, n)
            g = grad_hinge_one(X[i], y[i], w)
            w -= lr * g
        loss_hist.append(loss_value(X, y, w))

    return w, loss_hist


# Мультистарт
def multistart_best(X, y, n_starts=10, lr=1e-2, epochs=5, seed=42):
    rng = np.random.default_rng(seed)
    best_w = None
    best_loss = np.inf

    for k in range(n_starts):
        w0 = rng.normal(0, 0.01, size=X.shape[1])
        w = w0.copy()

        n = len(X)
        for _ in range(epochs):
            for _ in range(n):
                i = rng.integers(0, n)
                g = grad_hinge_one(X[i], y[i], w)
                w -= lr * g

        L = loss_value(X, y, w)
        if L < best_loss:
            best_loss = L
            best_w = w.copy()

    return best_w, best_loss


# Метрики + сравнение со sklearn

def eval_metrics(X, y, w, name):
    y_pred = np.sign(X @ w)
    print(f"\n[{name}]")
    print(f"Accuracy : {accuracy_score(y, y_pred):.3f}")
    print(f"Precision: {precision_score(y, y_pred, pos_label=1):.3f}")
    print(f"Recall   : {recall_score(y, y_pred, pos_label=1):.3f}")
    print(f"F1       : {f1_score(y, y_pred, pos_label=1):.3f}")

def plot_losses(loss_hist, title, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()

def plot_ema(q_ema_hist, title, filename, step=10):
    plt.figure(figsize=(8, 4))
    plt.plot(q_ema_hist[::step])
    plt.title(title)
    plt.xlabel(f"SGD steps (/{step})")
    plt.ylabel("Q (EMA)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()

# MAIN: запускаем по порядку заданий
def main():
    X, y = load_data("lab1/data.csv")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    Xtr, Xte = standardize_train_test(Xtr, Xte)
    Xtr, Xte = add_bias(Xtr), add_bias(Xte)

    # --- ШАГ 1: базовый SGD (без всего)
    w1, loss1 = sgd_basic(Xtr, ytr, lr=1e-2, epochs=10)
    plot_losses(loss1, "Basic SGD loss", "sgd_basic_loss.png")
    eval_metrics(Xte, yte, w1, "STEP1: Basic SGD")
    plot_sorted_margins(Xte, yte, w1, "Basic SGD margins", "sgd_basic_margins.png")


    # --- ШАГ 2: добавили EMA (рекуррентная оценка)
    w2, loss2, q2 = sgd_with_ema(Xtr, ytr, lr=1e-2, epochs=10, ema_lambda=0.05)
    plot_losses(loss2, "SGD + EMA loss", "sgd_ema_loss.png")
    plot_ema(q2, "SGD + EMA (Q)", "sgd_ema_q.png")
    eval_metrics(Xte, yte, w2, "STEP2: SGD + EMA")

    # --- ШАГ 3: momentum
    w3, loss3 = sgd_with_momentum(Xtr, ytr, lr=1e-2, epochs=10, gamma=0.9)
    plot_losses(loss3, "SGD with Momentum", "sgd_momentum.png")
    eval_metrics(Xte, yte, w3, "STEP3: SGD + Momentum")

    # --- ШАГ 4: L2
    w4, loss4 = sgd_with_l2(Xtr, ytr, lr=1e-2, epochs=10, tau=1e-2)
    plot_losses(loss4, "SGD with L2", "sgd_l2.png")
    eval_metrics(Xte, yte, w4, "STEP4: SGD + L2")

    # --- ШАГ 5: fastest descent
    w5, loss5 = sgd_fastest(Xtr, ytr, epochs=10)
    plot_losses(loss5, "Fastest descent", "sgd_fastest.png")
    eval_metrics(Xte, yte, w5, "STEP5: Fastest Descent")

    # --- ШАГ 6: margin-abs sampling
    w6, loss6 = sgd_margin_sampling(Xtr, ytr, lr=1e-2, epochs=10)
    plot_losses(loss6, "Margin-based sampling", "sgd_margin_sampling.png")
    eval_metrics(Xte, yte, w6, "STEP6: Margin-abs sampling")

    # --- ШАГ 7: correlation init
    w7, loss7 = sgd_with_corr_init(Xtr, ytr, lr=1e-2, epochs=10)
    plot_losses(loss7, "Correlation init", "sgd_corr_init.png")
    eval_metrics(Xte, yte, w7, "STEP7: Corr init + SGD")

    # --- ШАГ 8: multistart
    w8, bestL = multistart_best(Xtr, ytr, n_starts=10, lr=1e-2, epochs=10)
    print(f"\n[STEP8: Multistart] best train loss={bestL:.6f}")
    eval_metrics(Xte, yte, w8, "STEP8: Multistart best")

    # --- Эталон: sklearn
    skl = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-2, learning_rate="constant",
                        eta0=1e-2, max_iter=2000, random_state=42)
    skl.fit(Xtr, ytr)
    y_pred = skl.predict(Xte)
    print("\n[SKLEARN REF]")
    print(f"Accuracy :  {accuracy_score(yte, y_pred):.3f}")
    print(f"Precision: {precision_score(yte, y_pred, pos_label=1):.3f}")
    print(f"Recall   :  {recall_score(yte, y_pred, pos_label=1):.3f}")
    print(f"F1       : {f1_score(yte, y_pred, pos_label=1):.3f}")


if __name__ == "__main__":
    main()
