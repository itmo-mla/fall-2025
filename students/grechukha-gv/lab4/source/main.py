import os
import time

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from comparison import (
    compare_pca_implementations,
    detailed_variance_comparison,
    print_comparison_results,
)
from data_preprocessing import load_and_preprocess_data
from pca import PCA, determine_effective_dimensionality, reconstruction_error
from visualization import plot_all_visualizations
from report_generator import generate_final_report


def main():
    print("\n" + "------ Лабораторная работа №4: PCA через сингулярное разложение ------")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "------ Загрузка и предобработка данных ------")
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print(f"\nРазмеры данных:")
    print(f"  Обучающая выборка: {X_train.shape}")
    print(f"  Тестовая выборка:  {X_test.shape}")
    print(f"  Количество признаков: {X_train.shape[1]}")
    
    print("\n" + "------ Обучение PCA ------")
    
    our_pca = PCA()
    our_pca.fit(X_train)
    
    sklearn_pca = SklearnPCA()
    sklearn_pca.fit(X_train)
    
    print("\n" + "------ Определение эффективной размерности ------")
    
    dim_info = determine_effective_dimensionality(our_pca)
    
    print(f"\nЭффективная размерность датасета:")
    print(f"  95% дисперсии объясняется {dim_info['n_components_95']} компонентами")
    print(f"  99% дисперсии объясняется {dim_info['n_components_99']} компонентами")
    print(f"  Критерий Марченко-Пастура: {dim_info['marchenko_pastur']} компонент")
    
    if dim_info['elbow_point'] is not None:
        print(f"  Точка 'локтя' (elbow method): компонента #{dim_info['elbow_point']}")
    
    print(f"\nОбъясненная дисперсия первыми компонентами:")
    for i in range(min(10, len(our_pca.explained_variance_ratio_))):
        cumsum = np.sum(our_pca.explained_variance_ratio_[:i+1])
        print(f"  PC1-PC{i+1}: {our_pca.explained_variance_ratio_[i]*100:6.2f}% "
              f"(накопленная: {cumsum*100:6.2f}%)")
    
    print("\n" + "------ Сравнение с эталонной реализацией ------")
    
    n_comp_compare = min(30, X_train.shape[1])
    comparison_results = compare_pca_implementations(our_pca, X_train, n_components=n_comp_compare)
    print_comparison_results(comparison_results)
    
    sklearn_pca_compare = SklearnPCA(n_components=n_comp_compare)
    sklearn_pca_compare.fit(X_train)
    detailed_variance_comparison(our_pca, sklearn_pca_compare, n_show=10)
    
    print("\n" + "------ Анализ качества восстановления ------")
    
    test_components = [2, 5, 10, 20, dim_info['n_components_95'], dim_info['n_components_99']]
    test_components = sorted(list(set([c for c in test_components if c <= X_train.shape[1]])))
    
    print(f"\nОшибка восстановления (MSE):")
    print(f"{'Компоненты':>12} | {'MSE':>12} | {'Сохранено дисперсии':>20}")
    print("-" * 50)
    
    for n_comp in test_components:
        pca_temp = PCA(n_components=n_comp)
        X_transformed = pca_temp.fit_transform(X_train)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)
        
        mse = reconstruction_error(X_train, X_reconstructed)
        var_explained = np.sum(pca_temp.explained_variance_ratio_)
        
        print(f"{n_comp:12d} | {mse:12.6f} | {var_explained*100:19.2f}%")
    
    print("\n" + "------ Создание визуализаций ------")
    
    plot_all_visualizations(X_train, y_train, our_pca, sklearn_pca, results_dir)
    
    print("\n" + "------ Сохранение результатов ------")
    
    results_file = os.path.join(results_dir, 'final_results.txt')
    generate_final_report(
        results_file=results_file,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        our_pca=our_pca,
        dim_info=dim_info,
        comparison_results=comparison_results,
        test_components=test_components
    )
    
    print(f"\nРезультаты сохранены в: {results_file}")
    
    print("\n" + "------ Лабораторная работа выполнена успешно! ------")
    print(f"\nВсе результаты сохранены в директории: {results_dir}/")
    print("\nСозданные файлы:")
    print("  - scree_plot.png")
    print("  - cumulative_variance.png")
    print("  - pca_projection_2d.png")
    print("  - reconstruction_error.png")
    print("  - component_heatmap.png")
    print("  - final_results.txt")
    print("\n")


if __name__ == '__main__':
    main()
