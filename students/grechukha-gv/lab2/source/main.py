import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from comparison import (
    compare_with_sklearn,
    compare_with_without_prototypes,
    plot_comparison_bar,
    plot_prototype_comparison,
    print_confusion_matrix_detailed,
    calculate_metrics,
    analyze_errors,
    visualize_prototype_selection_process,
    compare_kernels,
    plot_kernel_comparison
)
from data_preprocessing import (
    X_train_array,
    X_test_array,
    y_train_array,
    y_test_array,
    visualize_data_pca,
    visualize_data_tsne,
    apply_smote
)
from knn import KNNParzenWindowEfficient
from loo_validation import (
    select_optimal_k,
    plot_loo_errors,
    analyze_k_sensitivity
)
from prototype_selection import (
    condensed_nearest_neighbor,
    stolp_algorithm,
    visualize_prototypes_pca,
    compare_prototype_methods,
    ccv_prototype_selection,
    plot_ccv_history
)
from utils import setup_logging, timer, measure_inference_time
from report_generator import generate_final_report


if __name__ == '__main__':
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    logger = setup_logging(os.path.join(results_dir, 'experiment_log.txt'))
    logger.info("------ Лабораторная работа №2: Метрическая классификация (KNN) ------")
    
    print("\n" + "------ Визуализация данных ------")
    
    # PCA визуализация
    pca_save_path = os.path.join(results_dir, 'pca_visualization.png')
    with timer("PCA визуализация", logger):
        visualize_data_pca(
            X_train_array, y_train_array,
            title="PCA визуализация обучающей выборки (Bank Marketing Dataset)",
            save_path=pca_save_path
        )
    
    # t-SNE визуализация
    tsne_save_path = os.path.join(results_dir, 'tsne_visualization.png')
    with timer("t-SNE визуализация", logger):
        visualize_data_tsne(
            X_train_array, y_train_array,
            title="t-SNE визуализация обучающей выборки (Bank Marketing Dataset)",
            save_path=tsne_save_path,
            perplexity=30
        )
    
    print(f"\nРазмер обучающей выборки: {len(X_train_array)} объектов")
    print(f"Размер тестовой выборки: {len(X_test_array)} объектов")
    print(f"Количество признаков: {X_train_array.shape[1]}")
    
    # Применяем SMOTE для балансировки классов
    print("\n" + "------ Балансировка классов ------")
    X_train_balanced, y_train_balanced = apply_smote(
        X_train_array, y_train_array,
        sampling_strategy='auto',
        k_neighbors=5,
        random_state=42
    )
    
    print("\n" + "------ Подбор параметра k ------")
    
    sample_size = min(1000, len(X_train_balanced))
    print(f"\nИспользуем подвыборку из {sample_size} объектов для LOO валидации (сбалансированная выборка)")
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_balanced), sample_size, replace=False)
    X_train_sample = X_train_balanced[sample_indices]
    y_train_sample = y_train_balanced[sample_indices]
    
    # Подбираем k с использованием F1-score для класса 1 (те, кто принимает условия)
    k_range = range(1, 31)
    with timer("LOO валидация для подбора k с метрикой F1-score", logger):
        loo_results = select_optimal_k(
            X_train_sample, y_train_sample,
            k_range=k_range,
            use_efficient=False,
            verbose=True,
            metric='f1',
            class_weights='balanced',
            pos_label=1
        )
    
    optimal_k = loo_results['optimal_k']
    logger.info(f"LOO: Оптимальное k={optimal_k}, F1-score={loo_results['optimal_score']:.4f}")
    
    # Анализируем чувствительность
    # Для F1-метрики используем инверсию для совместимости с analyze_k_sensitivity
    sensitivity_stats = analyze_k_sensitivity(
        loo_results['k_values'],
        loo_results['loo_errors']
    )
    
    # Строим график LOO (для F1-score - показываем как метрику)
    loo_plot_path = os.path.join(results_dir, 'loo_risk.png')
    if loo_results.get('metric') == 'f1':
        plt.figure(figsize=(12, 6))
        plt.plot(loo_results['k_values'], loo_results['loo_scores'], 'b-o', 
                linewidth=2, markersize=5, alpha=0.7, label='F1-score класса 1')
        optimal_idx = loo_results['k_values'].index(optimal_k)
        optimal_score = loo_results['loo_scores'][optimal_idx]
        plt.plot(optimal_k, optimal_score, 'r*', markersize=20, 
                label=f'Оптимальное k={optimal_k}')
        plt.xlabel('Количество соседей (k)', fontsize=12)
        plt.ylabel('F1-score для класса 1', fontsize=12)
        plt.title('Подбор параметра k по F1-score класса 1 (LOO)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.annotate(
            f'k={optimal_k}\nF1={optimal_score:.4f}',
            xy=(optimal_k, optimal_score),
            xytext=(optimal_k + len(loo_results['k_values']) * 0.1, optimal_score - 0.02),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        plt.tight_layout()
        plt.savefig(loo_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График LOO F1-score сохранен в: {loo_plot_path}")
    else:
        plot_loo_errors(
            loo_results['k_values'],
            loo_results['loo_errors'],
            optimal_k,
            save_path=loo_plot_path
        )
    
    print("\n" + "------ Сравнение ядер ------")
    
    # Сравниваем ядра на подвыборке (используем сбалансированные данные)
    kernel_sample_size = min(3000, len(X_train_balanced))
    kernel_sample_indices = np.random.choice(len(X_train_balanced), kernel_sample_size, replace=False)
    X_train_kernel_sample = X_train_balanced[kernel_sample_indices]
    y_train_kernel_sample = y_train_balanced[kernel_sample_indices]
    
    with timer("Сравнение ядер", logger):
        kernel_results = compare_kernels(
            X_train_kernel_sample, y_train_kernel_sample,
            X_test_array, y_test_array,
            k_range=range(1, 21),
            kernels=['gaussian', 'rectangular', 'triangular', 'epanechnikov']
        )
    
    # Строим график сравнения ядер
    kernel_comparison_path = os.path.join(results_dir, 'kernel_comparison.png')
    plot_kernel_comparison(kernel_results, range(1, 21), save_path=kernel_comparison_path)
    
    # Выбираем лучшее ядро
    best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['best_accuracy'])
    best_kernel_acc = kernel_results[best_kernel]['best_accuracy']
    logger.info(f"Лучшее ядро: {best_kernel} с accuracy={best_kernel_acc:.4f}")
    print(f"\nЛучшее ядро: {best_kernel}")
    
    print("\n" + "------ Обучение KNN на полной выборке ------")
    
    print(f"\nИспользуем оптимальное k={optimal_k} и лучшее ядро: {best_kernel}")
    print("Обучаем на сбалансированных данных с взвешенным голосованием")
    
    knn_optimal = KNNParzenWindowEfficient(
        k=optimal_k, 
        kernel=best_kernel,
        class_weights='balanced'  # Используем взвешенное голосование
    )
    
    with timer(f"Обучение KNN (k={optimal_k}, kernel={best_kernel}, balanced)", logger):
        knn_optimal.fit(X_train_balanced, y_train_balanced)
    
    print("Предсказание на тестовой выборке...")
    with timer("Предсказание на тестовой выборке", logger):
        y_pred_test = knn_optimal.predict(X_test_array)
    
    inference_stats = measure_inference_time(knn_optimal, X_test_array, n_runs=3)
    logger.info(f"Среднее время предсказания: {inference_stats['mean_time']:.3f}s")
    logger.info(f"Время на объект: {inference_stats['time_per_sample_ms']:.2f}ms")
    logger.info(f"Throughput: {inference_stats['throughput']:.1f} объектов/сек")
    
    test_metrics = calculate_metrics(
        y_test_array, y_pred_test,
        label=f"KNN (k={optimal_k}, kernel={best_kernel}) на тестовой выборке"
    )
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    print_confusion_matrix_detailed(
        y_test_array, y_pred_test,
        title="Confusion Matrix (KNN на полной выборке)"
    )
    
    # Анализ ошибок
    error_analysis_path = os.path.join(results_dir, 'error_analysis.png')
    analyze_errors(
        X_test_array, y_test_array, y_pred_test,
        title="Анализ ошибок классификации (Лучшая модель)",
        save_path=error_analysis_path
    )
    
    print("\n" + "------ Сравнение с sklearn ------")
    
    sklearn_comparison = compare_with_sklearn(
        X_train_array, y_train_array,
        X_test_array, y_test_array,
        k=optimal_k
    )
    
    comparison_plot_path = os.path.join(results_dir, 'comparison.png')
    plot_comparison_bar(sklearn_comparison, save_path=comparison_plot_path)
    
    print("\n" + "------ Алгоритм отбора эталонов ------")
    prototype_sample_size = min(2000, len(X_train_balanced))
    print(f"\nИспользуем подвыборку из {prototype_sample_size} объектов для отбора эталонов (сбалансированная)")
    
    prototype_indices_sample = np.random.choice(len(X_train_balanced), prototype_sample_size, replace=False)
    X_train_proto_sample = X_train_balanced[prototype_indices_sample]
    y_train_proto_sample = y_train_balanced[prototype_indices_sample]
    
    # CNN алгоритм
    print("\nАлгоритм Condensed Nearest Neighbor (CNN)")
    with timer("CNN отбор эталонов", logger):
        cnn_result = condensed_nearest_neighbor(
            X_train_proto_sample, y_train_proto_sample,
            k=optimal_k,
            max_iterations=20,
            verbose=True
        )
    logger.info(f"CNN: {cnn_result['n_prototypes']} эталонов, сжатие {cnn_result['compression_ratio']:.2%}")
    
    # Визуализация эталонов CNN
    prototype_selection_path = os.path.join(results_dir, 'prototype_selection.png')
    visualize_prototypes_pca(
        X_train_proto_sample, y_train_proto_sample,
        cnn_result['indices'],
        title="Визуализация эталонов CNN (PCA)",
        save_path=prototype_selection_path
    )
    
    # STOLP алгоритм
    print("\nАлгоритм STOLP")
    with timer("STOLP отбор эталонов", logger):
        stolp_result = stolp_algorithm(
            X_train_proto_sample, y_train_proto_sample,
            k=optimal_k,
            threshold=0.0,
            verbose=True
        )
    logger.info(f"STOLP: {stolp_result['n_prototypes']} эталонов, сжатие {stolp_result['compression_ratio']:.2%}")
    
    # CCV алгоритм
    print("\nАлгоритм CCV (Complete Cross-Validation)")
    with timer("CCV отбор эталонов", logger):
        ccv_result = ccv_prototype_selection(
            X_train_proto_sample, y_train_proto_sample,
            k=3,
            max_candidates=20,
            max_iterations=100,
            verbose=True
        )
    logger.info(f"CCV: {ccv_result['n_prototypes']} эталонов, сжатие {ccv_result['compression_ratio']:.2%}")
    
    # Визуализация эталонов STOLP (дополнительная)
    stolp_vis_path = os.path.join(results_dir, 'prototypes_stolp.png')
    visualize_prototypes_pca(
        X_train_proto_sample, y_train_proto_sample,
        stolp_result['indices'],
        title="Визуализация эталонов STOLP (PCA)",
        save_path=stolp_vis_path
    )
    
    # Визуализация CCV history
    if 'ccv_history' in ccv_result and len(ccv_result['ccv_history']) > 0:
        ccv_history_path = os.path.join(results_dir, 'ccv_history.png')
        plot_ccv_history(ccv_result['ccv_history'], save_path=ccv_history_path)
    
    # Визуализация эталонов CCV
    ccv_vis_path = os.path.join(results_dir, 'prototypes_ccv.png')
    visualize_prototypes_pca(
        X_train_proto_sample, y_train_proto_sample,
        ccv_result['indices'],
        title="Визуализация эталонов CCV (PCA)",
        save_path=ccv_vis_path
    )
    
    print("\n" + "------ Сравнение с/без отбора эталонов ------")
    
    prototype_comparison = compare_with_without_prototypes(
        X_train_proto_sample, y_train_proto_sample,
        X_test_array, y_test_array,
        cnn_result['indices'],
        k=optimal_k
    )
    
    proto_comparison_plot_path = os.path.join(results_dir, 'accuracy_with_prototypes.png')
    plot_prototype_comparison(
        prototype_comparison['full'],
        prototype_comparison['prototypes'],
        prototype_comparison['compression_ratio'],
        save_path=proto_comparison_plot_path
    )
    
    print("\n" + "------ Генерация отчета ------")
    
    results_file = os.path.join(results_dir, 'final_results.txt')
    generate_final_report(
        results_file=results_file,
        X_train_array=X_train_balanced,
        X_test_array=X_test_array,
        y_train_array=y_train_balanced,
        y_test_array=y_test_array,
        k_range=k_range,
        sample_size=sample_size,
        optimal_k=optimal_k,
        loo_results=loo_results,
        sensitivity_stats=sensitivity_stats,
        test_metrics=test_metrics,
        sklearn_comparison=sklearn_comparison,
        prototype_sample_size=prototype_sample_size,
        cnn_result=cnn_result,
        stolp_result=stolp_result,
        prototype_comparison=prototype_comparison
    )
    
    print(f"\nРезультаты сохранены в: {results_file}")
    print("\nГрафики сохранены в results/")
    