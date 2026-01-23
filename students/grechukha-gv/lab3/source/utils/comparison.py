import time

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC


def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def analyze_support_vectors(svm, X, y):
    print("\n" + "------ Детальный анализ опорных векторов ------")
    
    n_total = len(X)
    n_sv = len(svm.support_vectors)
    
    print(f"\nОбщая статистика:")
    print(f"  Количество опорных векторов: {n_sv}")
    print(f"  Доля от обучающей выборки: {100*n_sv/n_total:.2f}%")
    
    print(f"\nСтатистика по λ:")
    print(f"  Максимальное λ: {np.max(svm.support_vector_lambdas):.6f}")
    print(f"  Минимальное λ: {np.min(svm.support_vector_lambdas):.6f}")
    print(f"  Среднее λ: {np.mean(svm.support_vector_lambdas):.6f}")
    print(f"  Медиана λ: {np.median(svm.support_vector_lambdas):.6f}")
    print(f"  Стд. отклонение λ: {np.std(svm.support_vector_lambdas):.6f}")
    
    # Опорные векторы на границе margin (0 < λ < C)
    epsilon = 1e-5
    margin_svs = (svm.support_vector_lambdas > epsilon) & (svm.support_vector_lambdas < svm.C - epsilon)
    boundary_svs = svm.support_vector_lambdas >= svm.C - epsilon
    
    print(f"\nКатегории опорных векторов:")
    print(f"  На границе margin (0 < λ < C): {np.sum(margin_svs)} ({100*np.sum(margin_svs)/n_sv:.1f}%)")
    print(f"  На границе C (λ ≈ C): {np.sum(boundary_svs)} ({100*np.sum(boundary_svs)/n_sv:.1f}%)")
    
    sv_class_neg = np.sum(svm.support_vector_labels == -1)
    sv_class_pos = np.sum(svm.support_vector_labels == 1)
    
    print(f"\nРаспределение опорных векторов по классам:")
    print(f"  Класс -1: {sv_class_neg} ({100*sv_class_neg/n_sv:.1f}%)")
    print(f"  Класс +1: {sv_class_pos} ({100*sv_class_pos/n_sv:.1f}%)")
    
    # Распределение в исходной выборке
    total_neg = np.sum(y == -1)
    total_pos = np.sum(y == 1)
    
    print(f"\nДоля объектов, ставших опорными векторами:")
    print(f"  Из класса -1: {100*sv_class_neg/total_neg:.1f}%")
    print(f"  Из класса +1: {100*sv_class_pos/total_pos:.1f}%")
    
    print("-"*70)


def compare_with_sklearn(our_svm, X_train, y_train, X_test, y_test, kernel='linear', **kernel_params):
    print("\n" + f"------ Сравнение с sklearn (ядро: {kernel}) ------")
    
    print("\n[Собственная реализация]")
    our_predictions = our_svm.predict(X_test)
    
    our_metrics = evaluate_model(y_test, our_predictions)
    
    print(f"  Количество опорных векторов: {len(our_svm.support_vectors)}")
    print(f"  Accuracy:  {our_metrics['accuracy']:.4f}")
    print(f"  Precision: {our_metrics['precision']:.4f}")
    print(f"  Recall:    {our_metrics['recall']:.4f}")
    print(f"  F1-score:  {our_metrics['f1']:.4f}")
    
    print("\n[sklearn.svm.SVC]")
    
    if kernel == 'linear':
        sklearn_kernel = 'linear'
        sklearn_params = {}
    elif kernel == 'rbf':
        sklearn_kernel = 'rbf'
        gamma = kernel_params.get('gamma', 1.0)
        sklearn_params = {'gamma': gamma}
    elif kernel == 'polynomial':
        sklearn_kernel = 'poly'
        degree = kernel_params.get('degree', 3)
        gamma = kernel_params.get('gamma', 1.0)
        coef0 = kernel_params.get('coef0', 1.0)
        sklearn_params = {'degree': degree, 'gamma': gamma, 'coef0': coef0}
    else:
        raise ValueError(f"Неизвестное ядро: {kernel}")
    
    sklearn_svm = SVC(C=our_svm.C, kernel=sklearn_kernel, **sklearn_params)
    
    sklearn_svm.fit(X_train, y_train)
    sklearn_predictions = sklearn_svm.predict(X_test)
    
    sklearn_metrics = evaluate_model(y_test, sklearn_predictions)
    
    print(f"  Количество опорных векторов: {len(sklearn_svm.support_vectors_)}")
    print(f"  Accuracy:  {sklearn_metrics['accuracy']:.4f}")
    print(f"  Precision: {sklearn_metrics['precision']:.4f}")
    print(f"  Recall:    {sklearn_metrics['recall']:.4f}")
    print(f"  F1-score:  {sklearn_metrics['f1']:.4f}")
    
    print("\n[Разница (Собственная реализация - sklearn)]")
    print(f"  Δ Accuracy:  {our_metrics['accuracy'] - sklearn_metrics['accuracy']:+.4f}")
    print(f"  Δ Precision: {our_metrics['precision'] - sklearn_metrics['precision']:+.4f}")
    print(f"  Δ Recall:    {our_metrics['recall'] - sklearn_metrics['recall']:+.4f}")
    print(f"  Δ F1-score:  {our_metrics['f1'] - sklearn_metrics['f1']:+.4f}")
    print(f"  Δ Support Vectors: {len(our_svm.support_vectors) - len(sklearn_svm.support_vectors_):+d}")
    
    print("\n[Матрица ошибок - Собственная реализация]")
    print_confusion_matrix(our_metrics['confusion_matrix'])
    
    print("\n[Матрица ошибок - sklearn]")
    print_confusion_matrix(sklearn_metrics['confusion_matrix'])
    
    print("-"*70)
    
    return {
        'our': {
            'metrics': our_metrics,
            'n_support_vectors': len(our_svm.support_vectors)
        },
        'sklearn': {
            'metrics': sklearn_metrics,
            'n_support_vectors': len(sklearn_svm.support_vectors_)
        }
    }


def print_confusion_matrix(cm):
    print("                 Predicted")
    print("               -1        +1")
    print(f"Actual  -1   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"        +1   {cm[1,0]:5d}    {cm[1,1]:5d}")


def format_metrics_table(results_list):
    lines = []
    lines.append("\n" + "------ Сводная таблица результатов ------")
    
    header = f"{'Модель':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'SV':>8}"
    lines.append(header)
    lines.append("-"*90)
    
    for result in results_list:
        name = result['name']
        m = result['metrics']
        n_sv = result.get('n_sv', 'N/A')
        
        if isinstance(n_sv, int):
            sv_str = f"{n_sv:>8d}"
        else:
            sv_str = f"{n_sv:>8}"
        
        line = f"{name:<25} {m['accuracy']:10.4f} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1']:10.4f} {sv_str}"
        lines.append(line)
    
    lines.append("-"*90)
    
    return "\n".join(lines)


def save_comparison_results(results, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        if isinstance(results, dict):
            f.write("СРАВНЕНИЕ СОБСТВЕННОЙ РЕАЛИЗАЦИИ SVM С SKLEARN\n")
            f.write("="*70 + "\n\n")
            
            f.write("СОБСТВЕННАЯ РЕАЛИЗАЦИЯ:\n")
            f.write("-"*70 + "\n")
            our = results['our']
            f.write(f"  Accuracy:  {our['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {our['metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {our['metrics']['recall']:.4f}\n")
            f.write(f"  F1-score:  {our['metrics']['f1']:.4f}\n")
            f.write(f"  Опорные векторы: {our['n_support_vectors']}\n\n")
            
            f.write("SKLEARN.SVM.SVC:\n")
            f.write("-"*70 + "\n")
            sklearn = results['sklearn']
            f.write(f"  Accuracy:  {sklearn['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {sklearn['metrics']['precision']:.4f}\n")
            f.write(f"  Recall:    {sklearn['metrics']['recall']:.4f}\n")
            f.write(f"  F1-score:  {sklearn['metrics']['f1']:.4f}\n")
            f.write(f"  Опорные векторы: {sklearn['n_support_vectors']}\n\n")
            
            f.write("РАЗНИЦА (Собственная реализация - sklearn):\n")
            f.write("-"*70 + "\n")
            f.write(f"  Δ Accuracy:  {our['metrics']['accuracy'] - sklearn['metrics']['accuracy']:+.4f}\n")
            f.write(f"  Δ Precision: {our['metrics']['precision'] - sklearn['metrics']['precision']:+.4f}\n")
            f.write(f"  Δ Recall:    {our['metrics']['recall'] - sklearn['metrics']['recall']:+.4f}\n")
            f.write(f"  Δ F1-score:  {our['metrics']['f1'] - sklearn['metrics']['f1']:+.4f}\n")
        
        elif isinstance(results, list):
            table = format_metrics_table(results)
            f.write(table)
            f.write("\n")
