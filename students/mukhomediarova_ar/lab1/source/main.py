import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    # Запуск как пакет: python -m students.mukhomediarova_ar.lab1.source.main
    from .data_utils import (
        load_dataset,
        standardize_features,
        train_val_test_split,
    )
    from .linear_classifier import (
        SGDConfig,
        accuracy_score,
        add_bias_column,
        baseline_least_squares,
        classification_error,
        confusion_matrix,
        full_loss_and_grad,
        initialize_weights_correlation,
        initialize_weights_random,
        margins,
        quadratic_margin_loss_vectorized,
        sgd_train,
        steepest_gradient_descent,
        predict,
        multi_start_sgd,
    )
except ImportError:
    # Запуск напрямую: python students/mukhomediarova_ar/lab1/source/main.py
    from data_utils import (
        load_dataset,
        standardize_features,
        train_val_test_split,
    )
    from linear_classifier import (
        SGDConfig,
        accuracy_score,
        add_bias_column,
        baseline_least_squares,
        classification_error,
        confusion_matrix,
        full_loss_and_grad,
        initialize_weights_correlation,
        initialize_weights_random,
        margins,
        quadratic_margin_loss_vectorized,
        sgd_train,
        steepest_gradient_descent,
        predict,
        multi_start_sgd,
    )


def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_margins_histogram(margin_values: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(margin_values, bins=40, edgecolor="black", alpha=0.7)
    plt.xlabel("Отступ M_i")
    plt.ylabel("Частота")
    plt.title(title)
    plt.grid(alpha=0.3)
    out_path = _results_dir() / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Гистограмма отступов сохранена в {out_path}")


def plot_loss_curves(
    histories: List[Tuple[str, List[float]]],
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(9, 6))
    for label, hist in histories:
        plt.plot(hist, label=label)
    plt.xlabel("Эпоха")
    plt.ylabel("Средняя квадратичная потеря по отступу")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path = _results_dir() / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"График потерь сохранён в {out_path}")


def main() -> None:
    # 1. Датасет
    x, y = load_dataset()
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = train_val_test_split(x, y)

    x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)

    # Добавляем bias-координату
    x_train_b = add_bias_column(x_train)
    x_val_b = add_bias_column(x_val)
    x_test_b = add_bias_column(x_test)

    print(f"Train: {x_train_b.shape[0]}, Val: {x_val_b.shape[0]}, Test: {x_test_b.shape[0]}")

    # 2. Отступы для случайной инициализации и их анализ
    w0_random = initialize_weights_random(x_train_b.shape[1])
    m_train_init = margins(w0_random, x_train_b, y_train)
    print(
        f"Отступы до обучения: min={m_train_init.min():.3f}, "
        f"max={m_train_init.max():.3f}, mean={m_train_init.mean():.3f}"
    )
    plot_margins_histogram(
        m_train_init,
        "Распределение отступов до обучения",
        "margins_before_training.png",
    )

    # 3. Обучение SGD с инерцией и L2‑регуляризацией (инициализация по корреляции)
    w0_corr = initialize_weights_correlation(x_train, y_train)
    print("Инициализация весов по корреляции завершена.")

    base_config = SGDConfig(
        learning_rate=0.02,
        n_epochs=80,
        batch_size=16,
        alpha_l2=0.01,
        momentum=0.9,
        use_margin_sampling=False,
        ema_alpha=0.05,
        random_state=0,
    )

    w_sgd_corr, history_sgd_corr, ema_value = sgd_train(
        x_train_b,
        y_train,
        w0_corr,
        base_config,
    )

    m_train_sgd = margins(w_sgd_corr, x_train_b, y_train)
    m_val_sgd = margins(w_sgd_corr, x_val_b, y_val)
    print(
        f"После SGD+momentum+L2: train error={np.mean(m_train_sgd < 0):.3f}, "
        f"val error={np.mean(m_val_sgd < 0):.3f}"
    )
    print(f"Рекуррентная оценка функционала качества (EMA) после обучения: {ema_value:.4f}")

    # 4. Мультистарт со случайной инициализацией
    multi_config = SGDConfig(
        learning_rate=0.02,
        n_epochs=60,
        batch_size=16,
        alpha_l2=0.01,
        momentum=0.9,
        use_margin_sampling=False,
        ema_alpha=None,
    )
    w_multi, best_train_err = multi_start_sgd(
        x_train_b,
        y_train,
        n_starts=5,
        base_config=multi_config,
    )
    m_val_multi = margins(w_multi, x_val_b, y_val)
    print(
        f"Мультистарт: лучшая train error={best_train_err:.3f}, "
        f"val error={np.mean(m_val_multi < 0):.3f}"
    )

    # 5. Предъявление объектов по модулю отступа
    margin_sampling_config = SGDConfig(
        learning_rate=0.02,
        n_epochs=80,
        batch_size=16,
        alpha_l2=0.01,
        momentum=0.9,
        use_margin_sampling=True,
        ema_alpha=None,
        random_state=1,
    )
    w_margin, history_margin, _ = sgd_train(
        x_train_b,
        y_train,
        initialize_weights_random(x_train_b.shape[1]),
        margin_sampling_config,
    )
    m_val_margin = margins(w_margin, x_val_b, y_val)
    print(f"Margin‑sampling: val error={np.mean(m_val_margin < 0):.3f}")

    # 6. Скорейший градиентный спуск
    w0_steepest = initialize_weights_random(x_train_b.shape[1])
    w_steepest, history_steepest = steepest_gradient_descent(
        x_train_b,
        y_train,
        w0_steepest,
        alpha_l2=0.01,
        n_epochs=40,
        initial_step=1.0,
    )

    m_val_steepest = margins(w_steepest, x_val_b, y_val)
    print(f"Steepest GD: val error={np.mean(m_val_steepest < 0):.3f}")

    # 7. Сравнение кривых обучения
    histories = [
        ("SGD + momentum + L2 (corr init)", history_sgd_corr),
        ("SGD + margin‑sampling", history_margin),
        ("Steepest GD", history_steepest),
    ]
    plot_loss_curves(histories, "Сходимость различных вариантов обучения", "optimization_methods.png")

    # 8. Выбор лучшей модели по валидации
    candidates = {
        "SGD_corr": (w_sgd_corr, m_val_sgd),
        "Multi_start": (w_multi, m_val_multi),
        "Margin_sampling": (w_margin, m_val_margin),
        "Steepest_GD": (w_steepest, m_val_steepest),
    }
    best_name = min(candidates.keys(), key=lambda k: np.mean(candidates[k][1] < 0))
    w_best = candidates[best_name][0]
    print(f"Лучшая модель по валидационной ошибке: {best_name}")

    # 9. Финальная оценка на тестовой выборке
    y_pred_test = predict(w_best, x_test_b)
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nКачество лучшей модели на тесте:")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k}: {cm[k]:.4f}")

    # 10. Сравнение с эталонной линейной моделью (least squares)
    w_baseline = baseline_least_squares(x_train_b, y_train, alpha_l2=0.01)
    y_pred_baseline = predict(w_baseline, x_test_b)
    acc_best = accuracy_score(y_test, y_pred_test)
    acc_base = accuracy_score(y_test, y_pred_baseline)
    print(
        f"\nСравнение с эталонной моделью (least squares): "
        f"our_acc={acc_best:.4f}, baseline_acc={acc_base:.4f}"
    )

    # 11. График отступов для лучшей модели
    m_test_best = margins(w_best, x_test_b, y_test)
    plot_margins_histogram(
        m_test_best,
        "Распределение отступов на тестовой выборке (лучшая модель)",
        "margins_test_best.png",
    )


if __name__ == "__main__":
    main()

