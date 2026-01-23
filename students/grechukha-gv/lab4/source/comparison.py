import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error

from pca import PCA as OurPCA


def compare_pca_implementations(our_pca, X, n_components=None):
    if n_components is None:
        n_components = min(X.shape)
    
    sklearn_pca = SklearnPCA(n_components=n_components)
    sklearn_pca.fit(X)
    
    results = {}
    
    our_var_ratio = our_pca.explained_variance_ratio_[:n_components]
    sklearn_var_ratio = sklearn_pca.explained_variance_ratio_
    
    var_ratio_diff = np.abs(our_var_ratio - sklearn_var_ratio)
    results['explained_variance_ratio_diff'] = {
        'max': np.max(var_ratio_diff),
        'mean': np.mean(var_ratio_diff),
        'our': our_var_ratio,
        'sklearn': sklearn_var_ratio,
    }
    
    our_var = our_pca.explained_variance_[:n_components]
    sklearn_var = sklearn_pca.explained_variance_
    
    var_diff = np.abs(our_var - sklearn_var)
    results['explained_variance_diff'] = {
        'max': np.max(var_diff),
        'mean': np.mean(var_diff),
        'our': our_var,
        'sklearn': sklearn_var,
    }
    
    our_sv = our_pca.singular_values_[:n_components]
    sklearn_sv = sklearn_pca.singular_values_
    
    sv_diff = np.abs(our_sv - sklearn_sv)
    results['singular_values_diff'] = {
        'max': np.max(sv_diff),
        'mean': np.mean(sv_diff),
        'our': our_sv,
        'sklearn': sklearn_sv,
    }
    
    our_comp = our_pca.components_[:n_components]
    sklearn_comp = sklearn_pca.components_
    
    comp_diffs = []
    for i in range(n_components):
        diff_pos = np.linalg.norm(our_comp[i] - sklearn_comp[i])
        diff_neg = np.linalg.norm(our_comp[i] + sklearn_comp[i])
        comp_diffs.append(min(diff_pos, diff_neg))
    
    results['components_diff'] = {
        'max': np.max(comp_diffs),
        'mean': np.mean(comp_diffs),
        'component_norms': comp_diffs,
    }
    
    our_transformed = our_pca.transform(X)[:, :n_components]
    sklearn_transformed = sklearn_pca.transform(X)
    
    transform_diff = 0
    for i in range(n_components):
        diff_pos = np.linalg.norm(our_transformed[:, i] - sklearn_transformed[:, i])
        diff_neg = np.linalg.norm(our_transformed[:, i] + sklearn_transformed[:, i])
        transform_diff += min(diff_pos, diff_neg)
    
    transform_diff /= n_components
    
    results['transform_diff'] = transform_diff
    
    our_reconstructed = our_pca.inverse_transform(our_transformed)
    sklearn_reconstructed = sklearn_pca.inverse_transform(sklearn_transformed)
    
    reconstruction_diff = np.linalg.norm(our_reconstructed - sklearn_reconstructed)
    results['reconstruction_diff'] = reconstruction_diff
    
    our_mse = mean_squared_error(X, our_reconstructed)
    sklearn_mse = mean_squared_error(X, sklearn_reconstructed)
    
    results['reconstruction_mse'] = {
        'our': our_mse,
        'sklearn': sklearn_mse,
        'diff': abs(our_mse - sklearn_mse),
    }
    
    mean_diff = np.linalg.norm(our_pca.mean_ - sklearn_pca.mean_)
    results['mean_diff'] = mean_diff
    
    return results


def print_comparison_results(results):
    print("\n" + "------ Сравнение собственной реализации PCA с sklearn ------")
    
    print("\n1. EXPLAINED VARIANCE RATIO:")
    print(f"   Максимальное отличие: {results['explained_variance_ratio_diff']['max']:.2e}")
    print(f"   Среднее отличие:      {results['explained_variance_ratio_diff']['mean']:.2e}")
    
    print("\n2. EXPLAINED VARIANCE:")
    print(f"   Максимальное отличие: {results['explained_variance_diff']['max']:.2e}")
    print(f"   Среднее отличие:      {results['explained_variance_diff']['mean']:.2e}")
    
    print("\n3. SINGULAR VALUES:")
    print(f"   Максимальное отличие: {results['singular_values_diff']['max']:.2e}")
    print(f"   Среднее отличие:      {results['singular_values_diff']['mean']:.2e}")
    
    print("\n4. COMPONENTS (главные компоненты):")
    print(f"   Максимальное отличие: {results['components_diff']['max']:.2e}")
    print(f"   Среднее отличие:      {results['components_diff']['mean']:.2e}")
    
    print("\n5. TRANSFORM (проекция):")
    print(f"   Среднее отличие:      {results['transform_diff']:.2e}")
    
    print("\n6. INVERSE_TRANSFORM (восстановление):")
    print(f"   Отличие:              {results['reconstruction_diff']:.2e}")
    
    print("\n7. RECONSTRUCTION MSE:")
    print(f"   Собственная реализация: {results['reconstruction_mse']['our']:.6f}")
    print(f"   sklearn:                {results['reconstruction_mse']['sklearn']:.6f}")
    print(f"   Отличие:                {results['reconstruction_mse']['diff']:.2e}")
    
    print("\n8. MEAN (среднее по признакам):")
    print(f"   Отличие:              {results['mean_diff']:.2e}")
    
    print("\n" + "-"*70)
    
    threshold = 1e-10
    is_equivalent = (
        results['explained_variance_ratio_diff']['max'] < threshold and
        results['mean_diff'] < threshold
    )
    
    if is_equivalent:
        print("РЕАЛИЗАЦИИ ЭКВИВАЛЕНТНЫ (погрешность < 1e-10)")
    else:
        print("Реализации различаются, но это нормально из-за:")
        print("   - различий в численной точности")
        print("   - возможного изменения знака компонент в SVD")
    
    print("-"*70 + "\n")


def detailed_variance_comparison(our_pca, sklearn_pca, n_show=10):
    print("\nДЕТАЛЬНОЕ СРАВНЕНИЕ EXPLAINED VARIANCE RATIO:")
    print("-" * 70)
    print(f"{'№':>3} | {'Собственная':>12} | {'sklearn':>12} | {'Разница':>12} | {'%':>6}")
    print("-" * 70)
    
    n_show = min(n_show, len(our_pca.explained_variance_ratio_))
    
    for i in range(n_show):
        our_val = our_pca.explained_variance_ratio_[i]
        sklearn_val = sklearn_pca.explained_variance_ratio_[i]
        diff = abs(our_val - sklearn_val)
        percent = (our_val / sklearn_val - 1) * 100 if sklearn_val > 0 else 0
        
        print(f"{i+1:3d} | {our_val:12.8f} | {sklearn_val:12.8f} | {diff:12.2e} | {percent:6.2f}")
    
    print("-" * 70)
    
    our_cumsum = np.cumsum(our_pca.explained_variance_ratio_[:n_show])
    sklearn_cumsum = np.cumsum(sklearn_pca.explained_variance_ratio_[:n_show])
    
    print("\nНАКОПЛЕННАЯ ДИСПЕРСИЯ:")
    print("-" * 70)
    print(f"{'№':>3} | {'Собственная':>12} | {'sklearn':>12} | {'Разница':>12}")
    print("-" * 70)
    
    for i in range(min(5, n_show)):
        diff = abs(our_cumsum[i] - sklearn_cumsum[i])
        print(f"{i+1:3d} | {our_cumsum[i]:12.8f} | {sklearn_cumsum[i]:12.8f} | {diff:12.2e}")
    
    print("-" * 70 + "\n")


if __name__ == '__main__':
    print("Тест сравнения PCA реализаций")
    print("-" * 50)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    our_pca = OurPCA(n_components=5)
    our_pca.fit(X)
    
    results = compare_pca_implementations(our_pca, X, n_components=5)
    print_comparison_results(results)
    
    sklearn_pca = SklearnPCA(n_components=5)
    sklearn_pca.fit(X)
    detailed_variance_comparison(our_pca, sklearn_pca, n_show=5)
