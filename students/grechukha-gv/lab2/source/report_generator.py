import numpy as np


def generate_final_report(
    results_file,
    X_train_array,
    X_test_array,
    y_train_array,
    y_test_array,
    k_range,
    sample_size,
    optimal_k,
    loo_results,
    sensitivity_stats,
    test_metrics,
    sklearn_comparison,
    prototype_sample_size,
    cnn_result,
    stolp_result,
    prototype_comparison
):
    """
    Генерирует финальный отчет о результатах экспериментов.
    
    Args:
        results_file: путь к файлу для сохранения отчета
        X_train_array: обучающая выборка
        X_test_array: тестовая выборка
        y_train_array: метки обучающей выборки
        y_test_array: метки тестовой выборки
        k_range: диапазон значений k
        sample_size: размер выборки для LOO
        optimal_k: оптимальное значение k
        loo_results: результаты LOO валидации
        sensitivity_stats: статистика чувствительности
        test_metrics: метрики на тестовой выборке
        sklearn_comparison: результаты сравнения с sklearn
        prototype_sample_size: размер выборки для отбора эталонов
        cnn_result: результаты CNN алгоритма
        stolp_result: результаты STOLP алгоритма
        prototype_comparison: результаты сравнения с/без эталонов
    """
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("╔" + "═"*78 + "╗\n")
        f.write("║" + " "*20 + "ЛАБОРАТОРНАЯ РАБОТА №2" + " "*36 + "║\n")
        f.write("║" + " "*15 + "МЕТРИЧЕСКАЯ КЛАССИФИКАЦИЯ (KNN)" + " "*32 + "║\n")
        f.write("╚" + "═"*78 + "╝\n\n")
        
        f.write("ДАТАСЕТ: Bank Marketing Dataset\n")
        f.write("─"*80 + "\n")
        f.write(f"Обучающая выборка: {len(X_train_array)} объектов\n")
        f.write(f"Тестовая выборка: {len(X_test_array)} объектов\n")
        f.write(f"Количество признаков: {X_train_array.shape[1]}\n")
        f.write(f"Распределение классов в train: {np.bincount(y_train_array)}\n")
        f.write(f"Распределение классов в test: {np.bincount(y_test_array)}\n\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("1. ПОДБОР ОПТИМАЛЬНОГО k (LOO КРОСС-ВАЛИДАЦИЯ)\n")
        f.write("═"*80 + "\n")
        f.write(f"Диапазон k: {min(k_range)} - {max(k_range)}\n")
        f.write(f"Размер выборки для LOO: {sample_size} объектов\n")
        f.write(f"ОПТИМАЛЬНОЕ k: {optimal_k}\n")
        f.write(f"LOO ошибка при k={optimal_k}: {loo_results['optimal_error']:.4f}\n\n")
        
        f.write("Топ-5 значений k:\n")
        for i, (k, err) in enumerate(zip(sensitivity_stats['top_5_k'], sensitivity_stats['top_5_errors']), 1):
            f.write(f"  {i}. k={k:2d} -> ошибка={err:.4f}\n")
        f.write("\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("2. РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ (k={})\n".format(optimal_k))
        f.write("═"*80 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1-score:  {test_metrics['f1']:.4f}\n\n")
        
        cm = test_metrics['confusion_matrix']
        f.write("Confusion Matrix:\n")
        f.write("                  Predicted\n")
        f.write("             Class 0    Class 1\n")
        f.write(f"Actual  0    {cm[0, 0]:6d}     {cm[0, 1]:6d}\n")
        f.write(f"        1    {cm[1, 0]:6d}     {cm[1, 1]:6d}\n\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("3. СРАВНЕНИЕ С sklearn.neighbors.KNeighborsClassifier\n")
        f.write("═"*80 + "\n")
        f.write(f"{'Модель':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-score':>12}\n")
        f.write("─"*80 + "\n")
        
        # Наша реализация
        our = sklearn_comparison['our']
        f.write(
            f"{'Наша (Парзен)':<25} "
            f"{our['accuracy']:>12.4f} {our['precision']:>12.4f} "
            f"{our['recall']:>12.4f} {our['f1']:>12.4f}\n"
        )
        
        # sklearn uniform
        sk_unif = sklearn_comparison['sklearn_uniform']
        f.write(
            f"{'sklearn (uniform)':<25} "
            f"{sk_unif['accuracy']:>12.4f} {sk_unif['precision']:>12.4f} "
            f"{sk_unif['recall']:>12.4f} {sk_unif['f1']:>12.4f}\n"
        )
        
        # sklearn distance
        sk_dist = sklearn_comparison['sklearn_distance']
        f.write(
            f"{'sklearn (distance)':<25} "
            f"{sk_dist['accuracy']:>12.4f} {sk_dist['precision']:>12.4f} "
            f"{sk_dist['recall']:>12.4f} {sk_dist['f1']:>12.4f}\n"
        )
        f.write("\n")
        
        acc_diff_uniform = sklearn_comparison['our']['accuracy'] - sklearn_comparison['sklearn_uniform']['accuracy']
        acc_diff_distance = sklearn_comparison['our']['accuracy'] - sklearn_comparison['sklearn_distance']['accuracy']
        f.write(f"Разница с sklearn (uniform): {acc_diff_uniform:+.4f}\n")
        f.write(f"Разница с sklearn (distance): {acc_diff_distance:+.4f}\n\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("4. АЛГОРИТМ ОТБОРА ЭТАЛОНОВ\n")
        f.write("═"*80 + "\n")
        f.write(f"Размер выборки для отбора: {prototype_sample_size} объектов\n\n")
        
        f.write("CNN (Condensed Nearest Neighbor):\n")
        f.write(f"  Количество эталонов: {cnn_result['n_prototypes']}\n")
        f.write(f"  Степень сжатия: {cnn_result['compression_ratio']:.2%}\n\n")
        
        f.write("STOLP:\n")
        f.write(f"  Количество эталонов: {stolp_result['n_prototypes']}\n")
        f.write(f"  Степень сжатия: {stolp_result['compression_ratio']:.2%}\n\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("5. СРАВНЕНИЕ KNN С/БЕЗ ОТБОРА ЭТАЛОНОВ (CNN)\n")
        f.write("═"*80 + "\n")
        f.write(f"{'Модель':<25} {'Размер':>12} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-score':>12}\n")
        f.write("─"*80 + "\n")
        
        # KNN на полной выборке
        full = prototype_comparison['full']
        f.write(
            f"{'KNN (полная)':<25} {prototype_sample_size:>12} "
            f"{full['accuracy']:>12.4f} {full['precision']:>12.4f} "
            f"{full['recall']:>12.4f} {full['f1']:>12.4f}\n"
        )
        
        # KNN на эталонах
        proto = prototype_comparison['prototypes']
        f.write(
            f"{'KNN (эталоны CNN)':<25} {cnn_result['n_prototypes']:>12} "
            f"{proto['accuracy']:>12.4f} {proto['precision']:>12.4f} "
            f"{proto['recall']:>12.4f} {proto['f1']:>12.4f}\n"
        )
        
        f.write("─"*80 + "\n")
        
        # Разница
        compression_str = f"{prototype_comparison['compression_ratio']:.1%}"
        acc_diff = prototype_comparison['accuracy_diff']
        f.write(
            f"{'Разница':<25} {compression_str:>12} {acc_diff:>+12.4f} "
            f"{'':>12} {'':>12} {'':>12}\n"
        )
        f.write("\n")
        
        f.write("\n" + "═"*80 + "\n")
        f.write("ВЫВОДЫ\n")
        f.write("═"*80 + "\n")
        f.write("1. Оптимальное значение k подобрано методом LOO кросс-валидации.\n")
        f.write("2. KNN с окном Парзена показал сопоставимые результаты со sklearn.\n")
        f.write("3. Алгоритмы отбора эталонов (CNN, STOLP) значительно сокращают размер\n")
        f.write("   обучающей выборки при минимальной потере качества.\n")
        f.write("4. CNN алгоритм достигает высокой степени сжатия.\n")
        f.write("═"*80 + "\n")
