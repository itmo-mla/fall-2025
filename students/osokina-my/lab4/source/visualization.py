import numpy as np
import matplotlib.pyplot as plt


def plot_effective_dimension_analysis(eigenvalues, E_values, effective_dim, pca_svd, epsilon=0.01):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # График 1: Собственные значения (спектр)
    axes[0, 0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axvline(x=effective_dim + 1, color='r', linestyle='--', linewidth=2,
                       label=f'Эффективная размерность = {effective_dim}')
    axes[0, 0].set_xlabel('Номер компоненты', fontsize=12)
    axes[0, 0].set_ylabel('Собственное значение (λ)', fontsize=12)
    axes[0, 0].set_title('Спектр собственных значений', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(1, len(eigenvalues) + 1))

    # График 2: Критерий крутого склона (E_m)
    axes[0, 1].plot(range(len(E_values)), E_values, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=epsilon, color='r', linestyle='--', linewidth=2, label=f'ε = {epsilon}')
    axes[0, 1].axvline(x=effective_dim, color='r', linestyle='--', linewidth=2,
                       label=f'Эффективная размерность = {effective_dim}')
    axes[0, 1].set_xlabel('m (количество компонент)', fontsize=12)
    axes[0, 1].set_ylabel('E_m', fontsize=12)
    axes[0, 1].set_title('Критерий крутого склона: E_m', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')  # Логарифмическая шкала для лучшей визуализации

    # График 3: Кумулятивная объясненная дисперсия
    cumulative_variance = np.cumsum(pca_svd.explained_variance_ratio_)
    axes[1, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                    'mo-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=effective_dim + 1, color='r', linestyle='--', linewidth=2,
                       label=f'Эффективная размерность = {effective_dim}')
    axes[1, 0].axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% дисперсии')
    axes[1, 0].set_xlabel('Количество компонент', fontsize=12)
    axes[1, 0].set_ylabel('Кумулятивная объясненная дисперсия', fontsize=12)
    axes[1, 0].set_title('Кумулятивная объясненная дисперсия', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_xticks(range(1, len(cumulative_variance) + 1))

    # График 4: Отношение объясненной дисперсии
    axes[1, 1].bar(range(1, len(pca_svd.explained_variance_ratio_) + 1),
                   pca_svd.explained_variance_ratio_,
                   color='c', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].axvline(x=effective_dim + 0.5, color='r', linestyle='--', linewidth=2,
                       label=f'Эффективная размерность = {effective_dim}')
    axes[1, 1].set_xlabel('Номер компоненты', fontsize=12)
    axes[1, 1].set_ylabel('Отношение объясненной дисперсии', fontsize=12)
    axes[1, 1].set_title('Вклад каждой компоненты в дисперсию', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend()
    axes[1, 1].set_xticks(range(1, len(pca_svd.explained_variance_ratio_) + 1))

    plt.tight_layout()
    plt.show()

    cumulative_variance = np.cumsum(pca_svd.explained_variance_ratio_)
    print(f"\nЭффективная размерность: {effective_dim}")
    print(f"Объясненная дисперсия при m={effective_dim}: {cumulative_variance[effective_dim]:.4f}")
    print(f"Значение E_{effective_dim} = {E_values[effective_dim]:.6f}")


def plot_pca_comparison(X_pca_svd, X_pca_sklearn):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # График для нашей реализации
    axes[0].scatter(X_pca_svd[:, 0], X_pca_svd[:, 1], alpha=0.5, s=20)
    axes[0].set_xlabel('Первая главная компонента', fontsize=12)
    axes[0].set_ylabel('Вторая главная компонента', fontsize=12)
    axes[0].set_title('Наша реализация PCA (SVD)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # График для sklearn
    axes[1].scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], alpha=0.5, s=20, color='orange')
    axes[1].set_xlabel('Первая главная компонента', fontsize=12)
    axes[1].set_ylabel('Вторая главная компонента', fontsize=12)
    axes[1].set_title('sklearn PCA', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
