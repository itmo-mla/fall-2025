import os

import matplotlib.pyplot as plt
import numpy as np

from comparison import (
    optimization_methods_fig,
    multi_start_training,
    compare_with_baseline,
    print_confusion_matrix,
    generate_comparison_table
)
from data_preprocessing import (
    X_train_array,
    X_val_array,
    X_test_array,
    y_train_array,
    y_val_array,
    y_test_array,
    visualize_data_pca
)
from linear_classifier import add_bias_term, initialize_weights
from margins import (
    calculate_all_margins,
    margins_plot,
    plot_margin_ranking,
    analyze_margins_by_class
)
from stochastic_gradient_descent import (
    sgd_with_momentum,
    sgd_with_reg,
    stochastic_gradient_descent,
    stochastic_gradient_descent_logistic,
    steepest_gradient_descent,
    margin_based_sampling,
    sgd_with_ema,
)


if __name__ == '__main__':
    
    # Создаем директорию для результатов
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "-"*10 + " ВИЗУАЛИЗАЦИЯ ДАННЫХ " + "-"*10)
    
    # PCA визуализация данных
    pca_save_path = os.path.join(results_dir, 'pca_visualization.png')
    visualize_data_pca(X_train_array, y_train_array, 
                      title="PCA визуализация обучающей выборки (Bank Marketing Dataset)",
                      save_path=pca_save_path)
    
    X_train_with_bias = add_bias_term(X_train_array)
    X_val_with_bias = add_bias_term(X_val_array)
    X_test_with_bias = add_bias_term(X_test_array)
    
    w_init = initialize_weights(X_train_with_bias.shape[1])
    
    #Вычисляем отступы для всех объектов обучающей выборки
    margins = calculate_all_margins(w_init, X_train_with_bias, y_train_array) 
    print(f"Вычислено отступов: {len(margins)}")
    print(f"Минимальный отступ: {np.min(margins):.2f}")
    print(f"Максимальный отступ: {np.max(margins):.2f}")
    print(f"Средний отступ: {np.mean(margins):.2f}")

    #Вычисляем долю отрицательных отступов (ошибок)
    negative_margins = margins[margins < 0]
    error_rate_pre_train = len(negative_margins) / len(margins)
    print(f"Доля объектов с отрицательным отступом (ошибка): {error_rate_pre_train:.2f}")
    
    margins_plot(margins)
    
    print("\n" + "-"*5 + " ТЕСТИРУЕМ L2-РЕГУЛЯРИЗАЦИЮ И MOMENTUM " + "-"*5)

    def val_error_for_weights(w):
        m = calculate_all_margins(w, X_val_with_bias, y_val_array)
        return float(np.mean(m < 0))

    methods = {}

    # Базовые оптимизаторы и регуляризация
    methods["SGD базовый"] = lambda: stochastic_gradient_descent(
        X_train_with_bias,
        y_train_array,
        initialize_weights(X_train_with_bias.shape[1]),
        learning_rate=0.005,
        n_epochs=100,
        batch_size=32,
        plot=False,
        track_full_losses=True,
        X_train_full=X_train_with_bias,
        y_train_full=y_train_array,
        X_val_full=X_val_with_bias,
        y_val_full=y_val_array,
    )
    methods["SGD + L2"] = lambda: sgd_with_reg(
        X_train_with_bias,
        y_train_array,
        initialize_weights(X_train_with_bias.shape[1]),
        learning_rate=0.005,
        n_epochs=100,
        batch_size=32,
        reg_strength=0.01,
        plot=False,
        track_full_losses=True,
        X_train_full=X_train_with_bias,
        y_train_full=y_train_array,
        X_val_full=X_val_with_bias,
        y_val_full=y_val_array,
    )
    methods["SGD + Momentum"] = lambda: sgd_with_momentum(
        X_train_with_bias,
        y_train_array,
        initialize_weights(X_train_with_bias.shape[1]),
        learning_rate=0.005,
        n_epochs=100,
        batch_size=32,
        momentum=0.9,
        plot=False,
        track_full_losses=True,
        X_train_full=X_train_with_bias,
        y_train_full=y_train_array,
        X_val_full=X_val_with_bias,
        y_val_full=y_val_array,
    )

    # Инициализация через корреляцию
    methods["Correlation init + SGD"] = lambda: stochastic_gradient_descent(
        X_train_with_bias,
        y_train_array,
        initialize_weights(X_train_with_bias.shape[1], method='correlation', X=X_train_with_bias, y=y_train_array),
        learning_rate=0.005,
        n_epochs=100,
        batch_size=32,
        plot=False,
        track_full_losses=False,
    )

    # Мультистарт
    def _multistart():
        w_best, _best_err = multi_start_training(X_train_with_bias, y_train_array, n_starts=3, n_epochs=50, batch_size=32)
        return w_best, None
    methods["Multistart (3)"] = _multistart

    # Предъявление объектов по модулю отступа
    def _margin_sampling_uncertainty():
        w, loss_hist = margin_based_sampling(X_train_with_bias, y_train_array, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='uncertainty')
        return w, loss_hist
    def _margin_sampling_hard():
        w, loss_hist = margin_based_sampling(X_train_with_bias, y_train_array, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='hard_only')
        return w, loss_hist
    methods["Margin sampling (uncertainty)"] = _margin_sampling_uncertainty
    methods["Margin sampling (hard_only)"] = _margin_sampling_hard

    # Скорейший градиентный спуск
    def _steepest():
        w, loss_hist = steepest_gradient_descent(X_train_with_bias, y_train_array, n_epochs=60, batch_size=32)
        return w, loss_hist
    methods["Steepest GD"] = _steepest

    # SGD с рекуррентной оценкой функционала качества (EMA)
    def _sgd_ema():
        w, loss_hist, ema_loss = sgd_with_ema(X_train_with_bias, y_train_array, learning_rate=0.005, n_epochs=100, batch_size=32, lambda_ema=0.01, plot=False)
        return w, loss_hist, ema_loss
    methods["SGD + EMA"] = _sgd_ema

    # SGD с логистической функцией потерь
    methods["SGD (logistic loss)"] = lambda: stochastic_gradient_descent_logistic(
        X_train_with_bias,
        y_train_array,
        initialize_weights(X_train_with_bias.shape[1]),
        learning_rate=0.005,
        n_epochs=100,
        batch_size=32,
        plot=False,
        track_full_losses=True,
        X_train_full=X_train_with_bias,
        y_train_full=y_train_array,
        X_val_full=X_val_with_bias,
        y_val_full=y_val_array,
    )

    results = {}
    best_for_curves = None
    best_for_curves_val_error = float('inf')

    for name, runner in methods.items():
        print(f"\n--- {name} ---")
        out = runner()

        w_trained = out[0]
        loss_history = out[1] if len(out) > 1 else None
        train_full_loss_history = out[2] if len(out) > 3 else None
        val_full_loss_history = out[3] if len(out) > 3 else None

        v_err = val_error_for_weights(w_trained)
        results[name] = {
            "val_error": v_err,
            "loss_history": loss_history,
            "train_full_loss_history": train_full_loss_history,
            "val_full_loss_history": val_full_loss_history,
            "weights": w_trained,
        }
        if name == "SGD + EMA" and len(out) == 3:
            results[name]["ema_loss_last"] = float(out[2])

        print(f"Ошибка на валидации: {v_err:.3f}")
        
        if train_full_loss_history is not None and val_full_loss_history is not None and v_err < best_for_curves_val_error:
            best_for_curves_val_error = v_err
            best_for_curves = name

    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    optimization_methods_fig({
        k: (v["val_error"], v["loss_history"])
        for k, v in results.items()
        if v["loss_history"] is not None and k in {"SGD базовый", "SGD + L2", "SGD + Momentum"}
    })

    best_method_for_curves = best_for_curves if best_for_curves is not None else "SGD базовый"
    best_loss_history = results[best_method_for_curves]["loss_history"] or []
    best_train_full_loss_history = results[best_method_for_curves]["train_full_loss_history"] or []
    best_val_full_loss_history = results[best_method_for_curves]["val_full_loss_history"] or []
    plt.figure(figsize=(10, 6))
    plt.plot(best_loss_history, label='Batch/online loss (avg per epoch)')
    plt.plot(best_train_full_loss_history, label='Train loss (full dataset)')
    plt.plot(best_val_full_loss_history, label='Val loss (full dataset)')
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя потеря')
    plt.title(f'Сходимость SGD ({best_method_for_curves})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if best_train_full_loss_history and best_val_full_loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(best_train_full_loss_history, label='Train empirical risk', linewidth=2, color='blue')
        plt.plot(best_val_full_loss_history, label='Val empirical risk', linewidth=2, color='orange')
        plt.xlabel('Эпоха', fontsize=11)
        plt.ylabel('Эмпирический риск', fontsize=11)
        plt.title(f'Эмпирический риск ({best_method_for_curves})', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'empirical_risk.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nГрафик эмпирического риска сохранен: empirical_risk.png")
    
    print(f"\nОбучение завершено!")
    print(f"Размерность обученных весов: {w_trained.shape}")
    
    # Таблица сравнения всех методов
    print("\n")
    table_data = generate_comparison_table(results, X_train_with_bias, y_train_array, X_test_with_bias, y_test_array)

    # Финальная оценка на тестовой выборке (только после выбора лучшей модели)
    print("\n" + "-"*5 + " ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ " + "-"*5)

    # Выбираем лучшую модель по валидационной ошибке
    best_method = min(results.keys(), key=lambda x: results[x]["val_error"])
    print(f"Лучший метод по валидации: {best_method}")

    # Переобучаем лучшую модель на полном train+val наборе для финальной оценки
    X_train_val = np.vstack([X_train_with_bias, X_val_with_bias])
    y_train_val = np.concatenate([y_train_array, y_val_array])

    def retrain_on_train_val(method_name: str):
        if method_name == "SGD базовый":
            w0 = initialize_weights(X_train_val.shape[1])
            w, _ = stochastic_gradient_descent(X_train_val, y_train_val, w0, learning_rate=0.005, n_epochs=100, batch_size=32, plot=False)
            return w
        if method_name == "SGD + L2":
            w0 = initialize_weights(X_train_val.shape[1])
            w, _ = sgd_with_reg(X_train_val, y_train_val, w0, learning_rate=0.005, n_epochs=100, batch_size=32, reg_strength=0.01, plot=False)
            return w
        if method_name == "SGD + Momentum":
            w0 = initialize_weights(X_train_val.shape[1])
            w, _ = sgd_with_momentum(X_train_val, y_train_val, w0, learning_rate=0.005, n_epochs=100, batch_size=32, momentum=0.9, plot=False)
            return w
        if method_name == "Correlation init + SGD":
            w0 = initialize_weights(X_train_val.shape[1], method='correlation', X=X_train_val, y=y_train_val)
            w, _ = stochastic_gradient_descent(X_train_val, y_train_val, w0, learning_rate=0.005, n_epochs=100, batch_size=32, plot=False)
            return w
        if method_name == "Multistart (3)":
            w, _best_err = multi_start_training(X_train_val, y_train_val, n_starts=3, n_epochs=50, batch_size=32)
            return w
        if method_name == "Margin sampling (uncertainty)":
            w, _ = margin_based_sampling(X_train_val, y_train_val, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='uncertainty')
            return w
        if method_name == "Margin sampling (hard_only)":
            w, _ = margin_based_sampling(X_train_val, y_train_val, learning_rate=0.005, n_epochs=100, batch_size=32, strategy='hard_only')
            return w
        if method_name == "Steepest GD":
            w, _ = steepest_gradient_descent(X_train_val, y_train_val, n_epochs=60, batch_size=32)
            return w
        if method_name == "SGD + EMA":
            w, _loss_hist, _ema = sgd_with_ema(X_train_val, y_train_val, learning_rate=0.005, n_epochs=100, batch_size=32, lambda_ema=0.01, plot=False)
            return w
        if method_name == "SGD (logistic loss)":
            w0 = initialize_weights(X_train_val.shape[1])
            w, _ = stochastic_gradient_descent_logistic(X_train_val, y_train_val, w0, learning_rate=0.005, n_epochs=100, batch_size=32, plot=False)
            return w
        raise ValueError(f"Неизвестный метод: {method_name}")

    w_final = retrain_on_train_val(best_method)

    # Финальная оценка на тестовой выборке
    margins_test_final = calculate_all_margins(w_final, X_test_with_bias, y_test_array)
    error_rate_test = np.sum(margins_test_final < 0) / len(margins_test_final)

    print(f"\nФинальная ошибка на тесте: {error_rate_test:.3f}")
    print(f"Точность на тесте: {1 - error_rate_test:.3f}")
    
    # Детальный анализ отступов по классам на тесте
    analyze_margins_by_class(margins_test_final, y_test_array, 
                            title="Анализ отступов по классам (тестовая выборка)")

    plot_margin_ranking(margins_test_final, "Распределение отступов по рангу (тестовая выборка)")
    
    # Confusion Matrix для лучшей модели
    y_pred_test = np.sign(np.dot(X_test_with_bias, w_final))
    cm_metrics = print_confusion_matrix(y_test_array, y_pred_test, 
                                       title="Confusion Matrix (Лучшая модель на тесте)")

    # Сравнение с эталонной LogisticRegression
    our_acc, base_acc = compare_with_baseline(X_train_array, y_train_array, X_test_array, y_test_array, w_final)

    results_file = os.path.join(results_dir, 'final_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ЛАБОРАТОРНАЯ РАБОТА №1: ЛИНЕЙНАЯ КЛАССИФИКАЦИЯ\n")
        f.write("-"*60 + "\n\n")
        f.write("РАЗБИЕНИЕ ДАННЫХ:\n")
        f.write(f"  Обучающая выборка: {len(X_train_array)} объектов (70%)\n")
        f.write(f"  Валидационная выборка: {len(X_val_array)} объектов (15%)\n")
        f.write(f"  Тестовая выборка: {len(X_test_array)} объектов (15%)\n\n")

        f.write("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ:\n")
        for method, info in results.items():
            f.write(f"  {method}: ошибка на валидации = {info['val_error']:.4f}\n")
            if "ema_loss_last" in info:
                f.write(f"    EMA (последнее значение): {info['ema_loss_last']:.6f}\n")

        f.write(f"\nЛУЧШИЙ МЕТОД: {best_method}\n")
        f.write(f"ФИНАЛЬНАЯ ТОЧНОСТЬ НА ТЕСТЕ: {1 - error_rate_test:.4f}\n")
        f.write(f"ФИНАЛЬНАЯ ОШИБКА НА ТЕСТЕ: {error_rate_test:.4f}\n\n")

        f.write("СРАВНЕНИЕ С LogisticRegression (baseline):\n")
        f.write(f"  Наша accuracy: {our_acc:.4f}\n")
        f.write(f"  Baseline accuracy: {base_acc:.4f}\n")
        f.write(f"  Разница (ours - baseline): {our_acc - base_acc:+.4f}\n\n")

        f.write("CONFUSION MATRIX (лучшая модель на тесте):\n")
        f.write("                      Actual\n")
        f.write("                  Positive  Negative\n")
        f.write(f"Predict Positive     {cm_metrics['tp']:4d}      {cm_metrics['fp']:4d}\n")
        f.write(f"        Negative     {cm_metrics['fn']:4d}      {cm_metrics['tn']:4d}\n\n")
        f.write(f"  Precision: {cm_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {cm_metrics['recall']:.4f}\n")
        f.write(f"  F1-score:  {cm_metrics['f1']:.4f}\n\n")
        
        f.write("СТАТИСТИКА ОТСТУПОВ НА ТЕСТОВОЙ ВЫБОРКЕ:\n")
        f.write(f"  Минимальный отступ: {np.min(margins_test_final):.4f}\n")
        f.write(f"  Максимальный отступ: {np.max(margins_test_final):.4f}\n")
        f.write(f"  Средний отступ: {np.mean(margins_test_final):.4f}\n")
        f.write(f"  Доля уверенных предсказаний (|M| > 1): {np.mean(np.abs(margins_test_final) > 1):.1%}\n")
        f.write(f"  Доля пограничных предсказаний (0 ≤ |M| < 1): {np.mean((np.abs(margins_test_final) >= 0) & (np.abs(margins_test_final) < 1)):.1%}\n\n")
        
        f.write("СТАТИСТИКА ПО КЛАССАМ:\n")
        for label in [-1, 1]:
            class_margins = margins_test_final[y_test_array == label]
            f.write(f"\n  Класс {label}:\n")
            f.write(f"    Средний отступ: {np.mean(class_margins):.4f}\n")
            f.write(f"    Минимальный отступ: {np.min(class_margins):.4f}\n")
            f.write(f"    Доля ошибок: {np.mean(class_margins < 0):.1%}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("ТАБЛИЦА СРАВНЕНИЯ ВСЕХ МЕТОДОВ:\n")
        f.write("-"*80 + "\n")
        if table_data:
            f.write(f"{'Метод':<30} {'Val Err':>10} {'Test Acc':>10} {'Test Prec':>10} {'Test Rec':>10} {'Test F1':>10}\n")
            f.write("-"*80 + "\n")
            for row in table_data:
                f.write(f"{row['method']:<30} {row['val_error']:>10.4f} {row['test_acc']:>10.4f} {row['test_prec']:>10.4f} {row['test_rec']:>10.4f} {row['test_f1']:>10.4f}\n")
        f.write("-"*80 + "\n")

    print(f"\nРезультаты сохранены в: {results_file}")
