import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from data import load_and_preprocess_data, generate_linearly_separable_data, generate_circular_data
from models import SVM
from utils import (
    compare_with_sklearn,
    format_metrics_table,
    save_comparison_results,
    analyze_support_vectors,
    visualize_svm_2d,
    visualize_linear_svm_exact,
    visualize_support_vectors,
    visualize_with_tsne,
    plot_kernel_comparison,
    plot_pca_data
)


def test_on_synthetic_data(results_dir):
    print("\n" + "-"*70)
    print("ЧАСТЬ 1: Тестирование на синтетических данных")
    print("-"*70)
    
    print("\n1/2: Линейно разделимые данные (Linear kernel)")
    
    X_linear, y_linear = generate_linearly_separable_data(n_samples=200, n_features=2)
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
    print(f"Размер выборки: train={len(X_train)}, test={len(X_test)}")
    
    svm_linear = SVM(C=1.0, kernel='linear')
    svm_linear.fit(X_train, y_train, verbose=False)
    
    train_acc = svm_linear.score(X_train, y_train)
    test_acc = svm_linear.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Опорные векторы: {len(svm_linear.support_vectors)}")
    
    visualize_linear_svm_exact(
        svm_linear, X_train, y_train,
        title="Линейный SVM на синтетических данных",
        save_path=os.path.join(results_dir, 'synthetic_linear_exact.png')
    )
    
    comparison_linear_synth = compare_with_sklearn(
        svm_linear, X_train, y_train, X_test, y_test, kernel='linear'
    )
    
    print("\n2/2: Круговые данные (RBF kernel)")
    
    X_circular, y_circular = generate_circular_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X_circular, y_circular, test_size=0.2, random_state=42)
    print(f"Размер выборки: train={len(X_train)}, test={len(X_test)}")
    
    gamma = 2.0
    print(f"Обучение SVM с RBF ядром (gamma={gamma})...")
    svm_rbf = SVM(C=1.0, kernel='rbf', gamma=gamma)
    svm_rbf.fit(X_train, y_train, verbose=False)
    
    train_acc = svm_rbf.score(X_train, y_train)
    test_acc = svm_rbf.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Опорные векторы: {len(svm_rbf.support_vectors)}")
    
    visualize_svm_2d(
        svm_rbf, X_train, y_train,
        title=f"RBF SVM на круговых данных (γ={gamma})",
        save_path=os.path.join(results_dir, 'synthetic_circular_boundary.png')
    )
    
    comparison_rbf_synth = compare_with_sklearn(
        svm_rbf, X_train, y_train, X_test, y_test, kernel='rbf', gamma=gamma
    )
    
    return {
        'linear': comparison_linear_synth,
        'rbf': comparison_rbf_synth
    }


def test_on_real_data(results_dir):
    print("\n" + "-"*70)
    print("ЧАСТЬ 2: Тестирование на реальном датасете Bank Marketing")
    print("-"*70)
    
    print("\n1/5: Загрузка и предобработка данных")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    TRAIN_SAMPLE_SIZE = 800
    indices = np.random.choice(len(X_train), TRAIN_SAMPLE_SIZE, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    print(f"Подвыборка: {len(X_train_subset)} объектов, {X_train_subset.shape[1]} признаков")
    
    print("\n2/5: Визуализация данных")
    plot_pca_data(
        X_train_subset,
        y_train_subset,
        title="Bank Marketing Dataset (PCA 2D)",
        save_path=os.path.join(results_dir, 'pca_visualization.png')
    )
    
    ENABLE_TSNE = False
    
    if ENABLE_TSNE:
        visualize_with_tsne(
            X_train_subset,
            y_train_subset,
            title="Bank Marketing Dataset (t-SNE 2D)",
            save_path=os.path.join(results_dir, 'tsne_visualization.png'),
            perplexity=30
        )
    
    C = 1.0
    results = {}
    all_results = []
    
    print("\n3/5: Эксперимент 1 - Линейное ядро")
    
    svm_linear = SVM(C=C, kernel='linear')
    svm_linear.fit(X_train_subset, y_train_subset, verbose=False)
    
    test_acc_linear = svm_linear.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc_linear:.4f}")
    
    analyze_support_vectors(svm_linear, X_train_subset, y_train_subset)
    visualize_svm_2d(
        svm_linear, X_train_subset, y_train_subset,
        title=f"SVM с линейным ядром (C={C})",
        save_path=os.path.join(results_dir, 'decision_boundary_linear.png')
    )
    
    visualize_support_vectors(
        svm_linear, X_train_subset, y_train_subset,
        title="Анализ опорных векторов (линейное ядро)",
        save_path=os.path.join(results_dir, 'support_vectors_linear.png')
    )
    
    if ENABLE_TSNE:
        visualize_with_tsne(
            X_train_subset, y_train_subset, svm=svm_linear,
            title="t-SNE с опорными векторами (линейное ядро)",
            save_path=os.path.join(results_dir, 'tsne_sv_linear.png')
        )
    
    comparison_linear = compare_with_sklearn(
        svm_linear, X_train_subset, y_train_subset, X_test, y_test, kernel='linear'
    )
    results['linear'] = comparison_linear
    
    print("\n4/5: Эксперимент 2 - RBF (гауссово) ядро")
    
    gamma = 1.0 / (X_train_subset.shape[1] * X_train_subset.var())
    
    svm_rbf = SVM(C=C, kernel='rbf', gamma=gamma)
    svm_rbf.fit(X_train_subset, y_train_subset, verbose=False)
    
    test_acc_rbf = svm_rbf.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc_rbf:.4f}")
    
    analyze_support_vectors(svm_rbf, X_train_subset, y_train_subset)
    visualize_svm_2d(
        svm_rbf, X_train_subset, y_train_subset,
        title=f"SVM с RBF ядром (C={C}, γ={gamma:.6f})",
        save_path=os.path.join(results_dir, 'decision_boundary_rbf.png')
    )
    
    visualize_support_vectors(
        svm_rbf, X_train_subset, y_train_subset,
        title="Анализ опорных векторов (RBF ядро)",
        save_path=os.path.join(results_dir, 'support_vectors_rbf.png')
    )
    
    if ENABLE_TSNE:
        visualize_with_tsne(
            X_train_subset, y_train_subset, svm=svm_rbf,
            title="t-SNE с опорными векторами (RBF ядро)",
            save_path=os.path.join(results_dir, 'tsne_sv_rbf.png')
        )
    
    comparison_rbf = compare_with_sklearn(
        svm_rbf, X_train_subset, y_train_subset, X_test, y_test, kernel='rbf', gamma=gamma
    )
    results['rbf'] = comparison_rbf
    
    print("\n5/5: Эксперимент 3 - Полиномиальное ядро")
    
    degree = 2
    gamma_poly = 1.0 / (X_train_subset.shape[1] * X_train_subset.var())
    coef0 = 1.0
    print(f"Параметры: degree = {degree}, gamma = {gamma_poly:.6f}, coef0 = {coef0}")
    
    svm_poly = SVM(C=C, kernel='polynomial', degree=degree, gamma=gamma_poly, coef0=coef0)
    svm_poly.fit(X_train_subset, y_train_subset, verbose=False)
    
    test_acc_poly = svm_poly.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc_poly:.4f}")
    
    analyze_support_vectors(svm_poly, X_train_subset, y_train_subset)
    visualize_svm_2d(
        svm_poly, X_train_subset, y_train_subset,
        title=f"SVM с полиномиальным ядром (C={C}, d={degree})",
        save_path=os.path.join(results_dir, 'decision_boundary_poly.png')
    )
    
    visualize_support_vectors(
        svm_poly, X_train_subset, y_train_subset,
        title="Анализ опорных векторов (полиномиальное ядро)",
        save_path=os.path.join(results_dir, 'support_vectors_poly.png')
    )
    
    comparison_poly = compare_with_sklearn(
        svm_poly, X_train_subset, y_train_subset, X_test, y_test,
        kernel='polynomial', degree=degree, gamma=gamma_poly, coef0=coef0
    )
    results['poly'] = comparison_poly
    
    print("\n" + "-"*70)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("-"*70)
    kernel_results = [
        {
            'kernel': 'Linear',
            'accuracy': comparison_linear['our']['metrics']['accuracy'],
            'n_sv': comparison_linear['our']['n_support_vectors']
        },
        {
            'kernel': 'RBF',
            'accuracy': comparison_rbf['our']['metrics']['accuracy'],
            'n_sv': comparison_rbf['our']['n_support_vectors']
        },
        {
            'kernel': 'Polynomial',
            'accuracy': comparison_poly['our']['metrics']['accuracy'],
            'n_sv': comparison_poly['our']['n_support_vectors']
        }
    ]
    
    plot_kernel_comparison(
        kernel_results,
        save_path=os.path.join(results_dir, 'kernel_comparison.png')
    )
    
    all_results = [
        {'name': 'Собственная реализация (Linear)', 'metrics': comparison_linear['our']['metrics'], 
         'n_sv': comparison_linear['our']['n_support_vectors']},
        {'name': 'sklearn (Linear)', 'metrics': comparison_linear['sklearn']['metrics'], 
         'n_sv': comparison_linear['sklearn']['n_support_vectors']},
        {'name': 'Собственная реализация (RBF)', 'metrics': comparison_rbf['our']['metrics'], 
         'n_sv': comparison_rbf['our']['n_support_vectors']},
        {'name': 'sklearn (RBF)', 'metrics': comparison_rbf['sklearn']['metrics'], 
         'n_sv': comparison_rbf['sklearn']['n_support_vectors']},
        {'name': 'Собственная реализация (Polynomial)', 'metrics': comparison_poly['our']['metrics'], 
         'n_sv': comparison_poly['our']['n_support_vectors']},
        {'name': 'sklearn (Polynomial)', 'metrics': comparison_poly['sklearn']['metrics'], 
         'n_sv': comparison_poly['sklearn']['n_support_vectors']}
    ]
    
    print("\n" + "-"*70)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("-"*70)
    summary_table = format_metrics_table(all_results)
    print(summary_table)
    
    save_comparison_results(all_results, os.path.join(results_dir, 'final_results.txt'))
    
    with open(os.path.join(results_dir, 'final_results.txt'), 'a', encoding='utf-8') as f:
        f.write("\n\n")
        f.write("="*90 + "\n")
        f.write("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"Размер обучающей выборки: {len(X_train_subset)} объектов\n")
        f.write(f"Размер тестовой выборки: {len(X_test)} объектов\n")
        f.write(f"Количество признаков: {X_train_subset.shape[1]}\n\n")
        
        f.write(f"Параметр регуляризации C: {C}\n")
        f.write(f"Параметр RBF ядра gamma: {gamma:.6f}\n")
        f.write(f"Параметр полиномиального ядра degree: {degree}\n")
        f.write(f"Параметр полиномиального ядра coef0: {coef0}\n")
    
    return results


def main():
    print("\n" + "-"*70)
    print("Лабораторная работа 3: Support Vector Machine")
    print("-"*70)
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    synthetic_results = test_on_synthetic_data(results_dir)
    real_results = test_on_real_data(results_dir)
    
    print("\n" + "-"*70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("-"*70)
    print(f"\nРезультаты сохранены в: {results_dir}")
    print("\nСозданные файлы (14 файлов):")
    print("\nСинтетические данные:")
    print("  - synthetic_linear_exact.png")
    print("  - synthetic_circular_boundary.png")
    print("\nВизуализация данных:")
    print("  - pca_visualization.png")
    print("  - tsne_visualization.png")
    print("\nРазделяющие поверхности:")
    print("  - decision_boundary_linear.png")
    print("  - decision_boundary_rbf.png")
    print("  - decision_boundary_poly.png")
    print("\nАнализ опорных векторов:")
    print("  - support_vectors_linear.png")
    print("  - support_vectors_rbf.png")
    print("  - support_vectors_poly.png")
    print("\nt-SNE с опорными векторами:")
    print("  - tsne_sv_linear.png")
    print("  - tsne_sv_rbf.png")
    print("\nИтоговые результаты:")
    print("  - kernel_comparison.png")
    print("  - final_results.txt")
    print("\n" + "-"*70)


if __name__ == '__main__':
    main()
