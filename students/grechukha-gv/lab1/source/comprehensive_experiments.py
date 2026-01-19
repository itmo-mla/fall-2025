import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from comparison import compare_with_baseline, multi_start_training
from data_preprocessing import (
    X_train_array,
    X_val_array,
    X_test_array,
    y_train_array,
    y_val_array,
    y_test_array
)
from linear_classifier import add_bias_term, initialize_weights
from margins import calculate_all_margins, plot_margin_ranking, analyze_margins_detailed
from stochastic_gradient_descent import (
    stochastic_gradient_descent,
    sgd_with_reg,
    sgd_with_momentum,
    steepest_gradient_descent,
    margin_based_sampling
)


def run_comprehensive_experiments():
    """
    Запуск всех экспериментов согласно требованиям лабораторной работы
    """
    print("-"*15 + " КОМПЛЕКСНЫЕ ЭКСПЕРИМЕНТЫ ЛИНЕЙНОЙ КЛАССИФИКАЦИИ " + "-"*15)

    X_train_bias = add_bias_term(X_train_array)
    X_val_bias = add_bias_term(X_val_array)
    X_test_bias = add_bias_term(X_test_array)

    results_summary = {}

    # Инициализация через корреляцию
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 1: Инициализация через корреляцию " + "-"*5)

    w_corr = initialize_weights(X_train_bias.shape[1], method='correlation', X=X_train_bias, y=y_train_array)
    w_corr_trained, loss_history_corr = stochastic_gradient_descent(
        X_train_bias, y_train_array, w_corr, learning_rate=0.005, n_epochs=100, batch_size=32, plot=False
    )

    margins_corr_val = calculate_all_margins(w_corr_trained, X_val_bias, y_val_array)
    corr_val_accuracy = np.mean(margins_corr_val > 0)

    print(f"Точность на валидации: {corr_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_corr_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_corr_val <= 0):.4f}")

    results_summary['correlation_init'] = {
        'val_accuracy': corr_val_accuracy,
        'mean_margin': np.mean(margins_corr_val),
        'error_rate': np.mean(margins_corr_val <= 0)
    }

    # Мультистарт со случайной инициализацией
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 2: Мультистарт со случайной инициализацией " + "-"*5)

    best_w_multistart, _ = multi_start_training(X_train_bias, y_train_array, n_starts=3, n_epochs=50, batch_size=32)

    margins_multistart_val = calculate_all_margins(best_w_multistart, X_val_bias, y_val_array)
    multistart_val_accuracy = np.mean(margins_multistart_val > 0)

    print(f"Точность на валидации: {multistart_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_multistart_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_multistart_val <= 0):.4f}")

    results_summary['multistart'] = {
        'val_accuracy': multistart_val_accuracy,
        'mean_margin': np.mean(margins_multistart_val),
        'error_rate': np.mean(margins_multistart_val <= 0)
    }

    # Обучение с выбором объектов по отступам (Margin sampling - uncertainty)
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 3: Margin sampling (uncertainty) " + "-"*5)

    w_margin, loss_history_margin = margin_based_sampling(
        X_train_bias, y_train_array, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='uncertainty'
    )

    # Эксперимент 3b: Margin sampling с hard_only стратегией
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 3b: Margin sampling (hard_only) " + "-"*5)

    w_margin_hard, loss_history_margin_hard = margin_based_sampling(
        X_train_bias, y_train_array, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='hard_only'
    )

    margins_margin_val = calculate_all_margins(w_margin, X_val_bias, y_val_array)
    margin_val_accuracy = np.mean(margins_margin_val > 0)

    print(f"Точность на валидации: {margin_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_margin_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_margin_val <= 0):.4f}")

    margins_margin_hard_val = calculate_all_margins(w_margin_hard, X_val_bias, y_val_array)
    margin_hard_val_accuracy = np.mean(margins_margin_hard_val > 0)

    print(f"Точность на валидации: {margin_hard_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_margin_hard_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_margin_hard_val <= 0):.4f}")

    results_summary['margin_sampling'] = {
        'val_accuracy': margin_val_accuracy,
        'mean_margin': np.mean(margins_margin_val),
        'error_rate': np.mean(margins_margin_val <= 0)
    }

    results_summary['margin_sampling_hard'] = {
        'val_accuracy': margin_hard_val_accuracy,
        'mean_margin': np.mean(margins_margin_hard_val),
        'error_rate': np.mean(margins_margin_hard_val <= 0)
    }

    # Случайная инициализация
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 4: Случайная инициализация " + "-"*5)

    w_random = initialize_weights(X_train_bias.shape[1], method='random')
    w_random_trained, loss_history_random = stochastic_gradient_descent(
        X_train_bias, y_train_array, w_random, learning_rate=0.005, n_epochs=100, batch_size=32, plot=False
    )

    margins_random_val = calculate_all_margins(w_random_trained, X_val_bias, y_val_array)
    random_val_accuracy = np.mean(margins_random_val > 0)

    print(f"Точность на валидации: {random_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_random_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_random_val <= 0):.4f}")

    results_summary['random_init'] = {
        'val_accuracy': random_val_accuracy,
        'mean_margin': np.mean(margins_random_val),
        'error_rate': np.mean(margins_random_val <= 0)
    }

    # SGD с momentum
    print("\n" + "-"*5 + " ЭКСПЕРИМЕНТ 5: SGD + momentum " + "-"*5)

    w_momentum_trained, loss_history_momentum = sgd_with_momentum(
        X_train_bias, y_train_array, w_random, learning_rate=0.005, n_epochs=100, batch_size=32, momentum=0.9, plot=False
    )

    margins_momentum_val = calculate_all_margins(w_momentum_trained, X_val_bias, y_val_array)
    momentum_val_accuracy = np.mean(margins_momentum_val > 0)

    print(f"Точность на валидации: {momentum_val_accuracy:.4f}")
    print(f"Средний отступ (val): {np.mean(margins_momentum_val):.4f}")
    print(f"Доля ошибок (val): {np.mean(margins_momentum_val <= 0):.4f}")

    results_summary['momentum'] = {
        'val_accuracy': momentum_val_accuracy,
        'mean_margin': np.mean(margins_momentum_val),
        'error_rate': np.mean(margins_momentum_val <= 0)
    }

    # Определяем лучшую модель
    best_method = max(results_summary.keys(), key=lambda x: results_summary[x]['val_accuracy'])
    best_weights_map = {
        'correlation_init': w_corr_trained,
        'multistart': best_w_multistart,
        'margin_sampling': w_margin,
        'margin_sampling_hard': w_margin_hard,
        'random_init': w_random_trained,
        'momentum': w_momentum_trained,
    }
    best_w = best_weights_map[best_method]

    print("\n" + "-"*15 + " ВЫБОР ЛУЧШЕЙ МОДЕЛИ " + "-"*15)
    print(f"Лучший метод: {best_method} с точностью на валидации: {results_summary[best_method]['val_accuracy']:.4f}")

    # Детальный анализ лучшей модели
    print("\n" + "-"*5 + " ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ " + "-"*5)

    # Анализ на обучающей выборке
    margins_train_best = calculate_all_margins(best_w, X_train_bias, y_train_array)
    analyze_margins_detailed(margins_train_best, y_train_array, "Обучающая выборка - лучшая модель")

    # Анализ на валидационной выборке
    margins_val_best = calculate_all_margins(best_w, X_val_bias, y_val_array)
    analyze_margins_detailed(margins_val_best, y_val_array, "Валидационная выборка - лучшая модель")

    # Анализ на тестовой выборке
    margins_test_best = calculate_all_margins(best_w, X_test_bias, y_test_array)
    analyze_margins_detailed(margins_test_best, y_test_array, "Тестовая выборка - лучшая модель")

    # Визуализация отступов по рангу
    plot_margin_ranking(margins_test_best, "Распределение отступов по рангу (тестовая выборка)")

    # Сравнение с эталонной моделью
    our_accuracy, baseline_accuracy = compare_with_baseline(
        X_train_bias, y_train_array, X_test_bias, y_test_array, best_w
    )

    # Финальная таблица результатов
    print("\n" + "-"*15 + " ИТОГОВЫЕ РЕЗУЛЬТАТЫ " + "-"*15)

    print(f"{'Метод':<22}{'Val acc':>10}{'Mean margin':>14}{'Error rate':>12}")
    print("-" * 80)
    for method, metrics in results_summary.items():
        print(f"{method:<22}{metrics['val_accuracy']:>10.4f}{metrics['mean_margin']:>14.4f}{metrics['error_rate']:>12.4f}")

    print("-" * 80)
    print(f"{'Лучший метод':<22}{best_method:>10}{results_summary[best_method]['mean_margin']:>14.4f}{results_summary[best_method]['error_rate']:>12.4f}")

    print("\nМетрики лучшей модели на тестовой выборке:")
    test_predictions = np.sign(np.dot(X_test_bias, best_w))
    test_accuracy = accuracy_score(y_test_array, test_predictions)
    test_precision = precision_score(y_test_array, test_predictions)
    test_recall = recall_score(y_test_array, test_predictions)
    test_f1 = f1_score(y_test_array, test_predictions)

    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-score: {test_f1:.4f}")

    return results_summary, best_method, best_w

