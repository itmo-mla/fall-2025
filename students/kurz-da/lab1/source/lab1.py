from typing import Literal, Optional
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import SGDClassifier


class LinearClassifierConfig(BaseModel):
    """Конфигурация линейного классификатора"""
    # Базовые параметры
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6

    # Реализованные методы (ВКЛ/ВЫКЛ)
    use_momentum: bool = False  # Инерция
    momentum: float = 0.9

    use_regularization: bool = False  # L2 регуляризация
    reg_coefficient: float = 0.01

    use_stochastic: bool = False  # Стохастический градиент
    batch_size: int = 1

    use_recursive_q: bool = False  # Рекуррентная оценка Q
    lambda_forget: float = 0.05

    use_fastest_descent: bool = False  # Скорейший спуск

    use_margin_sampling: bool = False  # Предъявление по модулю отступа

    # Стратегии инициализации и обучения
    initialization: Literal["correlation", "random"] = "correlation"
    random_init_std: float = 0.01


class LinearClassifier:
    """
    Линейный классификатор с модульной архитектурой
    Каждая фича включается/выключается через конфиг
    """

    def __init__(self, config: LinearClassifierConfig = LinearClassifierConfig()):
        self.config = config
        self.w = None
        self.loss_history = []
        self.margin_history = []
        self.velocity = None  # Для инерции

    # ==========================================================================
    # РЕАЛИЗАЦИЯ: Вычисление отступа объекта
    # ==========================================================================
    def _margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        ВЫЧИСЛЕНИЕ ОТСТУПА ОБЪЕКТА
        M_i = y_i * (w·x_i) - отступ объекта x_i
        """
        return y * (X @ self.w)

    # ==========================================================================
    # РЕАЛИЗАЦИЯ: Вычисление градиента функции потерь
    # ==========================================================================
    def _loss_gradient(self, X: np.ndarray, y: np.ndarray, margin: np.ndarray) -> np.ndarray:
        """
        ВЫЧИСЛЕНИЕ ГРАДИЕНТА ФУНКЦИИ ПОТЕРЬ
        ∇Q(w) = -X^T @ ((1 - margin) * y) / n + λw (если регуляризация)
        """
        # Градиент квадратичной функции потерь
        grad = -X.T @ ((1 - margin) * y) / len(y)

        # ======================================================================
        # РЕАЛИЗАЦИЯ: L2 регуляризация
        # ======================================================================
        if self.config.use_regularization:
            grad += self.config.reg_coefficient * self.w

        return grad

    def _loss(self, margin: np.ndarray) -> float:
        """Функция потерь с опциональной L2 регуляризацией"""
        loss = 0.5 * np.mean((1 - margin) ** 2)

        if self.config.use_regularization:
            loss += 0.5 * self.config.reg_coefficient * np.sum(self.w ** 2)

        return loss

    # ==========================================================================
    # РЕАЛИЗАЦИЯ: Скорейший градиентный спуск
    # ==========================================================================
    def _fastest_descent_step(self, X: np.ndarray) -> float:
        """
        СКОРЕЙШИЙ ГРАДИЕНТНЫЙ СПУСК
        Оптимальный шаг для квадратичной функции потерь: h* = 1/||x||^2
        """
        if len(X.shape) == 1:
            step = 1.0 / (np.sum(X * X) + 1e-8)
        else:
            norms_sq = np.sum(X * X, axis=1)
            step = 1.0 / (np.mean(norms_sq) + 1e-8)

        return np.clip(step, 1e-8, 1.0)

    # ==========================================================================
    # РЕАЛИЗАЦИЯ: Предъявление объектов по модулю отступа
    # ==========================================================================
    def _sample_by_margin(self, X: np.ndarray, y: np.ndarray, margins: np.ndarray) -> tuple:
        """
        ПРЕДЪЯВЛЕНИЕ ОБЪЕКТОВ ПО МОДУЛЮ ОТСТУПА
        Вероятность выбора объекта обратно пропорциональна |margin|
        """
        weights = 1.0 / (np.abs(margins) + 1e-8)  # Обратно пропорционально отступу
        weights /= weights.sum()  # Нормализация
        indices = np.random.choice(len(X), size=self.config.batch_size, p=weights)
        return X[indices], y[indices]

    def _initialize_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Инициализация весов (корреляция или случайная)"""
        if self.config.initialization == "correlation":
            return np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
        else:
            return np.random.normal(0, self.config.random_init_std, X.shape[1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearClassifier":
        """
        ОБУЧЕНИЕ ЛИНЕЙНОГО КЛАССИФИКАТОРА
        Поддерживает разные стратегии через конфиг
        """
        # 1. Инициализация весов
        self.w = self._initialize_weights(X, y)

        # ======================================================================
        # РЕАЛИЗАЦИЯ: Метод стохастического градиентного спуска с инерцией
        # ======================================================================
        if self.config.use_momentum:
            self.velocity = np.zeros_like(self.w)  # инициализация "скорости" для инерции

        self.loss_history = []
        self.margin_history = []

        # Инициализация функционала качества
        current_margins = self._margin(X, y)
        Q = self._loss(current_margins)
        self.loss_history.append(Q)

        # ======================================================================
        # РЕАЛИЗАЦИЯ: Рекуррентная оценка функционала качества
        # ======================================================================
        if self.config.use_recursive_q:
            # Инициализация Q по случайному подмножеству
            init_size = min(100, len(X))  # размер выборки
            init_indices = np.random.choice(len(X), init_size, replace=False)
            init_margins = self._margin(X[init_indices], y[init_indices])
            Q = self._loss(init_margins)
            self.loss_history = [Q]

        for iteration in range(self.config.max_iterations):
            # ==================================================================
            # ВЫБОР РЕЖИМА ОБУЧЕНИЯ: стохастический или полный градиент
            # ==================================================================
            if self.config.use_stochastic:
                # СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК
                if self.config.use_margin_sampling:
                    # Предъявление по модулю отступа
                    current_margins = self._margin(X, y)
                    X_batch, y_batch = self._sample_by_margin(X, y, current_margins)
                else:
                    # Случайное предъявление
                    indices = np.random.choice(len(X), self.config.batch_size, replace=False)
                    X_batch, y_batch = X[indices], y[indices]
            else:
                # ПОЛНЫЙ ГРАДИЕНТНЫЙ СПУСК (вся выборка)
                X_batch, y_batch = X, y

            # Вычисление отступов и потерь
            margins_batch = self._margin(X_batch, y_batch)
            current_loss = self._loss(margins_batch)

            # ================================================================
            # РЕКУРРЕНТНАЯ ОЦЕНКА Q (если включена)
            # ================================================================
            if self.config.use_recursive_q:
                Q = self.config.lambda_forget * current_loss + (1 - self.config.lambda_forget) * Q
                self.loss_history.append(Q)
            else:
                self.loss_history.append(current_loss)

            # Вычисление градиента
            grad = self._loss_gradient(X_batch, y_batch, margins_batch)

            # ================================================================
            # ВЫБОР LEARNING RATE: обычный или скорейший спуск
            # ================================================================
            if self.config.use_fastest_descent:
                lr = self._fastest_descent_step(X_batch)
            else:
                lr = self.config.learning_rate

            # ================================================================
            # ОБНОВЛЕНИЕ ВЕСОВ: с инерцией или без
            # ================================================================
            if self.config.use_momentum:
                self.velocity = self.config.momentum * self.velocity - lr * grad
                w_new = self.w + self.velocity
            else:
                w_new = self.w - lr * grad

            # Критерий остановки
            if np.linalg.norm(w_new - self.w) < self.config.tolerance:
                break

            self.w = w_new

            # Сохранение отступов для анализа
            if iteration % 50 == 0:
                self.margin_history.append(self._margin(X, y).copy())

        # Финальные отступы
        self.margin_history.append(self._margin(X, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов"""
        return np.where(X @ self.w >= 0, 1, -1)


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЭКСПЕРИМЕНТОВ
# ==============================================================================

def multi_start_fit(config: LinearClassifierConfig, X: np.ndarray, y: np.ndarray,
                    n_restarts: int = 5) -> LinearClassifier:
    """
    ОБУЧЕНИЕ СО СЛУЧАЙНОЙ ИНИЦИАЛИЗАЦИЕЙ ВЕСОВ ЧЕРЕЗ МУЛЬТИСТАРТ
    """
    best_model = None
    best_loss = float('inf')

    for restart in range(n_restarts):
        current_config = config.model_copy()
        if restart > 0:
            current_config.initialization = "random"

        model = LinearClassifier(current_config)
        model.fit(X, y)

        current_loss = model.loss_history[-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best_model = model

    return best_model


def evaluate_model(model: LinearClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """Вычисление метрик качества"""
    predictions = model.predict(X)
    y_true = np.where(y == -1, 0, 1)
    y_pred = np.where(predictions == -1, 0, 1)

    return {
        'accuracy': f"{accuracy_score(y_true, y_pred):.3f}",
        'recall': f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
        'precision': f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
        'f1': f"{f1_score(y_true, y_pred, zero_division=0):.3f}"
    }


def visualize_loss_and_margins(model, X, y, title):
    """
    Визуализация лоссов и маржинов вместе
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График лоссов
    ax1.plot(model.loss_history)
    ax1.set_title(f'{title} - История потерь')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Функционал качества Q(w)')
    ax1.grid(True, alpha=0.3)

    # График маржинов
    margins = model._margin(X, y)
    sorted_margins = np.sort(margins)
    colors = ['red' if m < -0.5 else 'yellow' if m < 0.5 else 'green' for m in sorted_margins]

    ax2.scatter(range(len(sorted_margins)), sorted_margins, c=colors, alpha=0.6, s=20)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7)
    ax2.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Ранг объекта')
    ax2.set_ylabel('Отступ')
    ax2.set_title(f'{title} - Распределение отступов')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_experiments(X, y):
    """Запуск всех экспериментов согласно заданию"""

    print("=" * 60)
    print("ЭКСПЕРИМЕНТЫ ПО ЗАДАНИЮ")
    print("=" * 60)

    # 1. БАЗОВАЯ МОДЕЛЬ: обычный градиентный спуск
    print("\n1. ОБУЧИТЬ ЛИНЕЙНЫЙ КЛАССИФИКАТОР (базовый)")
    config1 = LinearClassifierConfig(
        use_stochastic=False,  # Полный градиент
        use_momentum=False,
        use_regularization=False,
        initialization="correlation"
    )
    model1 = LinearClassifier(config1)
    model1.fit(X, y)
    metrics1 = evaluate_model(model1, X, y)
    print(f"Метрики: {metrics1}")
    print(f"Финальный лосс: {model1.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model1, X, y, "Базовый классификатор")

    # 2. КОРРЕЛЯЦИОННАЯ ИНИЦИАЛИЗАЦИЯ
    print("\n2. ОБУЧИТЬ С ИНИЦИАЛИЗАЦИЕЙ ВЕСОВ ЧЕРЕЗ КОРРЕЛЯЦИЮ")
    config2 = LinearClassifierConfig(
        use_stochastic=False,
        initialization="correlation"
    )
    model2 = LinearClassifier(config2)
    model2.fit(X, y)
    metrics2 = evaluate_model(model2, X, y)
    print(f"Метрики: {metrics2}")
    print(f"Финальный лосс: {model2.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model2, X, y, "Инициализация через корреляцию")

    # 3. СЛУЧАЙНАЯ ИНИЦИАЛИЗАЦИЯ + МУЛЬТИСТАРТ
    print("\n3. ОБУЧИТЬ СО СЛУЧАЙНОЙ ИНИЦИАЛИЗАЦИЕЙ ВЕСОВ ЧЕРЕЗ МУЛЬТИСТАРТ")
    config3 = LinearClassifierConfig(
        use_stochastic=False,
        initialization="random"
    )
    model3 = multi_start_fit(config3, X, y, n_restarts=5)
    metrics3 = evaluate_model(model3, X, y)
    print(f"Метрики: {metrics3}")
    print(f"Финальный лосс: {model3.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model3, X, y, "Случайная инициализация + мультистарт")

    # 4. СЛУЧАЙНОЕ ПРЕДЪЯВЛЕНИЕ (стохастический режим)
    print("\n4. ОБУЧИТЬ СО СЛУЧАЙНЫМ ПРЕДЪЯВЛЕНИЕМ")
    config4 = LinearClassifierConfig(
        use_stochastic=True,  # Включаем стохастический режим
        use_margin_sampling=False,  # Случайные батчи
        batch_size=32
    )
    model4 = LinearClassifier(config4)
    model4.fit(X, y)
    metrics4 = evaluate_model(model4, X, y)
    print(f"Метрики: {metrics4}")
    print(f"Финальный лосс: {model4.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model4, X, y, "Случайное предъявление")

    # 5. ПРЕДЪЯВЛЕНИЕ ПО МОДУЛЮ ОТСТУПА
    print("\n5. ОБУЧИТЬ С ПРЕДЪЯВЛЕНИЕМ ОБЪЕКТОВ ПО МОДУЛЮ ОТСТУПА")
    config5 = LinearClassifierConfig(
        use_stochastic=True,  # Стохастический режим
        use_margin_sampling=True,  # Включаем выборку по отступам
        batch_size=32
    )
    model5 = LinearClassifier(config5)
    model5.fit(X, y)
    metrics5 = evaluate_model(model5, X, y)
    print(f"Метрики: {metrics5}")
    print(f"Финальный лосс: {model5.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model5, X, y, "Выборка по отступам")

    print("\n6. СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК С ИНЕРЦИЕЙ")
    config6 = LinearClassifierConfig(
        use_stochastic=True,  # Включаем стохастический режим
        use_momentum=True,  # ВКЛЮЧАЕМ ИНЕРЦИЮ
        momentum=0.9,  # Коэффициент инерции
        use_margin_sampling=False,  # Случайные батчи
        batch_size=32,
        use_recursive_q=True,  # Рекуррентная оценка Q для стохастического режима
        lambda_forget=0.05
    )
    model6 = LinearClassifier(config5)
    model6.fit(X, y)
    metrics6 = evaluate_model(model6, X, y)
    print(f"Метрики: {metrics6}")
    print(f"Финальный лосс: {model6.loss_history[-1]:.4f}")
    visualize_loss_and_margins(model6, X, y, "Стохастический спуск с инерцией")

    return {
        "Базовый": (model1, metrics1),
        "Корреляционная инициализация": (model2, metrics2),
        "Случайная инициализация (мультистарт)": (model3, metrics3),
        "Случайное предъявление": (model4, metrics4),
        "По модулю отступа": (model5, metrics5),
        "Стохастический град спуск с инерцией": (model6, metrics6)
    }


def compare_with_sklearn(X, y):
    """
    Сравнение с sklearn SGDClassifier
    """
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С SKLEARN SGDClassifier")
    print("=" * 60)

    # Обучаем SGDClassifier с квадратными потерями
    sk_model = SGDClassifier(
        loss='squared_hinge',  # Квадратичные потери как у нас
        penalty='l2',  # L2 регуляризация
        alpha=0.01,  # Коэффициент регуляризации
        learning_rate='constant',  # Постоянная скорость обучения
        eta0=0.01,  # Скорость обучения
        max_iter=1000,
        tol=1e-6,
        random_state=42
    )

    sk_model.fit(X, y)

    # Предсказания и метрики
    sk_pred = sk_model.predict(X)
    y_true = np.where(y == -1, 0, 1)
    y_pred = np.where(sk_pred == -1, 0, 1)

    sk_metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 3),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 3),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 3),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 3)
    }

    print(f"Sklearn SGDClassifier метрики: {sk_metrics}")

    # Визуализация отступов sklearn (decision function)
    sk_scores = sk_model.decision_function(X)

    plt.figure(figsize=(10, 5))
    sorted_scores = np.sort(sk_scores)
    colors = ['red' if s < -0.5 else 'yellow' if s < 0.5 else 'green' for s in sorted_scores]

    plt.scatter(range(len(sorted_scores)), sorted_scores, c=colors, alpha=0.6, s=20)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
    plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7)
    plt.xlabel('Ранг объекта')
    plt.ylabel('Decision Score')
    plt.title('Sklearn SGDClassifier - Распределение decision scores')
    plt.grid(True, alpha=0.3)
    plt.show()

    return sk_model, sk_metrics


# Использование
if __name__ == "__main__":
    # Запуск экспериментов с нашими моделями
    results = run_experiments(X_normalized, y)

    # Сравнение с sklearn
    sk_model, sk_metrics = compare_with_sklearn(X_normalized, y)