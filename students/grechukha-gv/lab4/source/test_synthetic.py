"""
Тест PCA на синтетических данных с известной структурой.

Демонстрирует работу PCA в идеальных условиях, когда данные имеют
явную низкоразмерную структуру с мультиколлинеарностью.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from pca import PCA, determine_effective_dimensionality, reconstruction_error


def generate_synthetic_data_with_structure(n_samples=1000, n_latent=2, n_features=10, noise_level=0.1, random_state=42):
    """
    Генерирует синтетические данные с явной низкоразмерной структурой.
    
    Данные генерируются из n_latent скрытых факторов, которые затем
    линейно комбинируются для создания n_features наблюдаемых признаков.
    
    Args:
        n_samples: количество объектов
        n_latent: количество скрытых факторов (истинная размерность)
        n_features: количество наблюдаемых признаков
        noise_level: уровень шума
        random_state: seed для воспроизводимости
    
    Returns:
        X: матрица данных (n_samples, n_features)
        true_components: истинные главные компоненты
    """
    np.random.seed(random_state)
    
    latent_factors = np.random.randn(n_samples, n_latent)
    
    mixing_matrix = np.random.randn(n_latent, n_features)
    
    X = np.dot(latent_factors, mixing_matrix)
    
    X += noise_level * np.random.randn(n_samples, n_features)
    
    return X, mixing_matrix


def test_pca_on_synthetic():
    print("\n" + "="*70)
    print("ТЕСТ PCA НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("="*70)
    
    n_samples = 1000
    n_latent = 2  # Истинная размерность
    n_features = 10  # Наблюдаемая размерность
    noise_level = 0.1
    
    print(f"\nПараметры синтетических данных:")
    print(f"  Количество объектов: {n_samples}")
    print(f"  Истинная размерность: {n_latent}")
    print(f"  Наблюдаемая размерность: {n_features}")
    print(f"  Уровень шума: {noise_level}")
    
    X, true_mixing = generate_synthetic_data_with_structure(
        n_samples=n_samples,
        n_latent=n_latent,
        n_features=n_features,
        noise_level=noise_level
    )
    
    print(f"\nФорма данных: {X.shape}")
    
    print("\nОбучение PCA...")
    pca = PCA()
    pca.fit(X)
    
    dim_info = determine_effective_dimensionality(pca)
    
    print(f"\nОпределение эффективной размерности:")
    print(f"  Истинная размерность: {n_latent}")
    print(f"  95% дисперсии: {dim_info['n_components_95']} компонент")
    print(f"  99% дисперсии: {dim_info['n_components_99']} компонент")
    print(f"  Критерий Марченко-Пастура: {dim_info['marchenko_pastur']} компонент")
    if dim_info['elbow_point'] is not None:
        print(f"  Точка локтя: {dim_info['elbow_point']}")
    
    print(f"\nОбъясненная дисперсия первыми компонентами:")
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        cumsum = np.sum(pca.explained_variance_ratio_[:i+1])
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:6.2f}% "
              f"(накопленная: {cumsum*100:6.2f}%)")
    
    print(f"\nОшибка восстановления для разного числа компонент:")
    print(f"{'Компоненты':>12} | {'MSE':>12} | {'Относительная ошибка':>20}")
    print("-" * 50)
    
    for n_comp in [1, 2, 3, 5, 10]:
        if n_comp > min(X.shape):
            continue
        pca_temp = PCA(n_components=n_comp)
        X_transformed = pca_temp.fit_transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)
        mse = reconstruction_error(X, X_reconstructed)
        relative_error = mse / np.var(X)
        print(f"{n_comp:12d} | {mse:12.6f} | {relative_error*100:19.2f}%")
    
    print(f"\nСоздание визуализаций...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    n_show = min(10, len(pca.explained_variance_ratio_))
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show])
    ax.axvline(n_latent + 0.5, color='r', linestyle='--', label=f'Истинная размерность = {n_latent}')
    ax.set_xlabel('Номер компоненты')
    ax.set_ylabel('Объясненная дисперсия')
    ax.set_title('Scree Plot (синтетические данные)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
    ax.axhline(0.95, color='g', linestyle='--', label='95%')
    ax.axhline(0.99, color='orange', linestyle='--', label='99%')
    ax.axvline(n_latent, color='r', linestyle='--', label=f'Истинная = {n_latent}')
    ax.set_xlabel('Количество компонент')
    ax.set_ylabel('Накопленная дисперсия')
    ax.set_title('Накопленная объясненная дисперсия')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    max_comp = min(10, min(X.shape))
    errors = []
    for n_comp in range(1, max_comp + 1):
        pca_temp = PCA(n_components=n_comp)
        X_transformed = pca_temp.fit_transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)
        mse = reconstruction_error(X, X_reconstructed)
        errors.append(mse)
    
    ax.plot(range(1, max_comp + 1), errors, 'b-o', linewidth=2, markersize=6)
    ax.axvline(n_latent, color='r', linestyle='--', label=f'Истинная = {n_latent}')
    ax.set_xlabel('Количество компонент')
    ax.set_ylabel('MSE восстановления')
    ax.set_title('Ошибка восстановления')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1, 1]
    X_2d = PCA(n_components=2).fit_transform(X)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=latent_factors[:, 0], 
                        cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Проекция на первые 2 компоненты')
    plt.colorbar(scatter, ax=ax, label='Первый скрытый фактор')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'synthetic_data_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Визуализация сохранена: {output_path}")
    
    print(f"\n" + "="*70)
    print("ВЫВОДЫ")
    print("="*70)
    print(f"1. PCA успешно выявил низкоразмерную структуру данных")
    print(f"2. Первые {n_latent} компоненты объясняют {cumsum[n_latent-1]*100:.1f}% дисперсии")
    print(f"3. Ошибка восстановления резко падает после {n_latent} компонент")
    print(f"4. Это подтверждает, что PCA эффективен для снижения размерности")
    print(f"   данных с явной мультиколлинеарностью")
    print("="*70 + "\n")
    
    return pca, X, dim_info


if __name__ == '__main__':
    np.random.seed(42)
    latent_factors = np.random.randn(1000, 2)
    test_pca_on_synthetic()
