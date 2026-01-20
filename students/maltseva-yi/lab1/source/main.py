import numpy as np
from linear_classifier import LinearClassifier
from utils import (
    load_and_preprocess_data,
    visualize_margins,
    visualize_training_history,
    multiclass_visualization,
    calculate_metrics
)
from experiments import ExperimentRunner


def main():

    print("\n1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("-"*40)
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("\n\n2. БАЗОВОЕ ОБУЧЕНИЕ ЛИНЕЙНОГО КЛАССИФИКАТОРА")
    print("-"*40)
    
    n_features = X_train.shape[1]
    
    # Используем логистическую потерю для стабильности
    classifier = LinearClassifier(
        n_features=n_features,
        learning_rate=0.01,
        reg_coef=0.001,
        momentum=0.9,
        random_state=42,
        loss_type='logistic'
    )
    
    # Обучение
    print("Начало обучения...")
    history = classifier.fit_sgd(
        X_train, y_train,
        n_epochs=50,
        batch_size=32,
        adaptive_lr=True,
        margin_selection=False,
        verbose=True
    )

    print("\n\n3. ВИЗУАЛИЗАЦИЯ ОТСТУПОВ")
    print("-"*40)
    
    visualize_margins(
        classifier, X_test, y_test,
        title="Распределение отступов на тестовой выборке",
        save_path="images/margins_distribution.png"
    )

    print("\n\n4. ИСТОРИЯ ОБУЧЕНИЯ")
    print("-"*40)
    
    visualize_training_history(
        history,
        save_path="images/training_history.png"
    )
    
    print("\n\n5. ВИЗУАЛИЗАЦИЯ РАЗДЕЛЯЮЩЕЙ ПОВЕРХНОСТИ")
    print("-"*40)
    
    multiclass_visualization(
        classifier, X_test, y_test,
        feature_indices=(0, 1),
        title="Разделяющая поверхность (первые два признака)",
        save_path="images/decision_boundary.png"
    )

    print("\n\n6. КОМПЛЕКСНЫЕ ЭКСПЕРИМЕНТЫ")
    print("-"*40)
    
    experiment_runner = ExperimentRunner(X_train, y_train, X_test, y_test)
    all_results = experiment_runner.run_comprehensive_experiment()

    print("\n\n7. СВОДКА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    best_model = all_results['multistart']['best_model']
    best_accuracy = all_results['multistart']['best_accuracy']
    
    print(f"\nЛучшая модель (мультистарт):")
    print(f"  Точность: {best_accuracy:.4f}")
    print(f"  Норма весов: {np.linalg.norm(best_model.weights):.4f}")
    print(f"  Смещение: {best_model.bias:.4f}")
    
    # Сравнение методов инициализации
    print(f"\nСравнение инициализаций:")
    init_results = all_results['initialization']
    for method_name, result in init_results.items():
        print(f"  {method_name}: точность = {result['metrics']['accuracy']:.4f}, "
              f"время = {result['training_time']:.2f} сек")
    
    # Сравнение стратегий оптимизации
    print(f"\nСравнение стратегий оптимизации:")
    strategies_results = all_results['strategies']
    best_strategy = max(
        strategies_results.items(),
        key=lambda x: x[1]['metrics']['accuracy']
    )
    print(f"  Лучшая стратегия: {best_strategy[0]}")
    print(f"  Точность: {best_strategy[1]['metrics']['accuracy']:.4f}")

    print("\n\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("="*70)
    
    results_table = []
    
    # Наша лучшая модель
    y_pred_best = best_model.predict(X_test)
    metrics_best = calculate_metrics(y_test, y_pred_best)
    results_table.append(("Наша лучшая модель", metrics_best))
    
    # Sklearn модель
    sklearn_comparison = all_results['sklearn_comparison']
    results_table.append(("Sklearn SGDClassifier", sklearn_comparison['sklearn_metrics']))
    
    # Вывод таблицы
    print(f"\n{'Метод':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-"*70)
    
    for method_name, metrics in results_table:
        print(f"{method_name:<30} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f}")
    
    return {
        'classifier': classifier,
        'best_model': best_model,
        'all_results': all_results
    }


if __name__ == "__main__":
    results = main()