import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from pca import PCA, reconstruction_error


def plot_scree(pca, save_path=None, max_components=None):
    if pca.explained_variance_ is None:
        raise ValueError("PCA не обучен")
    
    n_components = len(pca.explained_variance_)
    if max_components is not None:
        n_components = min(n_components, max_components)
    
    variance = pca.explained_variance_[:n_components]
    components = np.arange(1, n_components + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(components, variance, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Номер главной компоненты', fontsize=12)
    ax1.set_ylabel('Объясненная дисперсия', fontsize=12)
    ax1.set_title('Scree Plot: Объясненная дисперсия', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    variance_ratio = pca.explained_variance_ratio_[:n_components]
    ax2.bar(components, variance_ratio * 100, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Номер главной компоненты', fontsize=12)
    ax2.set_ylabel('Доля объясненной дисперсии (%)', fontsize=12)
    ax2.set_title('Scree Plot: Доля дисперсии', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scree plot сохранен: {save_path}")
    
    plt.close()


def plot_cumulative_variance(pca, save_path=None, thresholds=[0.95, 0.99]):
    if pca.explained_variance_ratio_ is None:
        raise ValueError("PCA не обучен")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(cumulative_variance) + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(components, cumulative_variance * 100, 
             linewidth=2, marker='o', markersize=4, color='steelblue')
    
    colors = ['red', 'orange', 'green']
    for i, threshold in enumerate(thresholds):
        n_comp = np.argmax(cumulative_variance >= threshold) + 1
        plt.axhline(y=threshold * 100, color=colors[i % len(colors)], 
                   linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'{threshold*100:.0f}% дисперсии: {n_comp} компонент')
        plt.axvline(x=n_comp, color=colors[i % len(colors)], 
                   linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Количество главных компонент', fontsize=12)
    plt.ylabel('Накопленная объясненная дисперсия (%)', fontsize=12)
    plt.title('Накопленная объясненная дисперсия', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    text_info = f"1 компонента: {cumulative_variance[0]*100:.2f}%\n"
    if len(cumulative_variance) >= 2:
        text_info += f"2 компоненты: {cumulative_variance[1]*100:.2f}%\n"
    if len(cumulative_variance) >= 5:
        text_info += f"5 компонент: {cumulative_variance[4]*100:.2f}%"
    
    plt.text(0.02, 0.98, text_info, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График накопленной дисперсии сохранен: {save_path}")
    
    plt.close()


def plot_pca_projection_2d(X, y, our_pca, sklearn_pca, save_path=None):
    """
    Сравнивает проекции данных на первые 2 компоненты.
    """
    X_our = our_pca.transform(X)[:, :2]
    X_sklearn = sklearn_pca.transform(X)[:, :2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_values = np.unique(y)
    is_classification = len(unique_values) <= 10
    
    if is_classification:
        colors_list = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))
        
        for i, cls in enumerate(unique_values):
            mask = y == cls
            ax1.scatter(X_our[mask, 0], X_our[mask, 1], 
                       c=[colors_list[i]], label=f'Класс {cls}', alpha=0.6, s=20, 
                       edgecolors='k', linewidth=0.3)
        
        for i, cls in enumerate(unique_values):
            mask = y == cls
            ax2.scatter(X_sklearn[mask, 0], X_sklearn[mask, 1], 
                       c=[colors_list[i]], label=f'Класс {cls}', alpha=0.6, s=20, 
                       edgecolors='k', linewidth=0.3)
        
        ax1.legend(fontsize=9, loc='best')
        ax2.legend(fontsize=9, loc='best')
    else:
        scatter1 = ax1.scatter(X_our[:, 0], X_our[:, 1], 
                              c=y, cmap='viridis', alpha=0.6, s=20, 
                              edgecolors='k', linewidth=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Целевая переменная')
        
        scatter2 = ax2.scatter(X_sklearn[:, 0], X_sklearn[:, 1], 
                              c=y, cmap='viridis', alpha=0.6, s=20, 
                              edgecolors='k', linewidth=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Целевая переменная')
    
    var1 = our_pca.explained_variance_ratio_[0] * 100
    var2 = our_pca.explained_variance_ratio_[1] * 100
    ax1.set_xlabel(f'PC1 ({var1:.2f}% дисперсии)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.2f}% дисперсии)', fontsize=11)
    ax1.set_title('Собственная реализация PCA', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    var1_sk = sklearn_pca.explained_variance_ratio_[0] * 100
    var2_sk = sklearn_pca.explained_variance_ratio_[1] * 100
    ax2.set_xlabel(f'PC1 ({var1_sk:.2f}% дисперсии)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({var2_sk:.2f}% дисперсии)', fontsize=11)
    ax2.set_title('sklearn.decomposition.PCA', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График проекций сохранен: {save_path}")
    
    plt.close()


def plot_reconstruction_error(X, max_components=None, save_path=None):
    if max_components is None:
        max_components = min(50, min(X.shape))
    
    errors = []
    components = []
    
    print(f"Вычисление ошибок восстановления для {max_components} компонент...")
    
    for n_comp in range(1, max_components + 1):
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        error = reconstruction_error(X, X_reconstructed)
        
        errors.append(error)
        components.append(n_comp)
        
        if n_comp % 10 == 0:
            print(f"  {n_comp} компонент: MSE = {error:.6f}")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(components, errors, linewidth=2, marker='o', markersize=4, color='darkblue')
    
    plt.xlabel('Количество главных компонент', fontsize=12)
    plt.ylabel('MSE восстановления', fontsize=12)
    plt.title('Ошибка восстановления от числа компонент', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    key_points = [1, 5, 10, 20] if max_components >= 20 else [1, 5, max_components]
    for kp in key_points:
        if kp <= len(errors):
            plt.annotate(f'{errors[kp-1]:.4f}', 
                        xy=(kp, errors[kp-1]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График ошибки восстановления сохранен: {save_path}")
    
    plt.close()


def plot_component_heatmap(pca, feature_names=None, n_components=5, save_path=None):
    if pca.components_ is None:
        raise ValueError("PCA не обучен")
    
    n_components = min(n_components, len(pca.components_))
    components = pca.components_[:n_components]
    
    n_features_show = min(20, components.shape[1])
    
    feature_importance = np.sum(np.abs(components), axis=0)
    top_features_idx = np.argsort(feature_importance)[-n_features_show:]
    
    components_subset = components[:, top_features_idx]
    
    if feature_names is not None:
        feature_labels = [feature_names[i] for i in top_features_idx]
    else:
        feature_labels = [f"F{i}" for i in top_features_idx]
    
    plt.figure(figsize=(12, 6))
    
    im = plt.imshow(components_subset, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    
    plt.colorbar(im, label='Вес признака')
    plt.xlabel('Признаки', fontsize=11)
    plt.ylabel('Главные компоненты', fontsize=11)
    plt.title(f'Веса топ-{n_features_show} признаков в главных компонентах', 
              fontsize=12, fontweight='bold')
    
    plt.yticks(range(n_components), [f'PC{i+1}' for i in range(n_components)])
    plt.xticks(range(n_features_show), feature_labels, rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Тепловая карта компонент сохранена: {save_path}")
    
    plt.close()


def plot_all_visualizations(X, y, our_pca, sklearn_pca, results_dir, feature_names=None):
    print("\n------ Создание визуализаций ------")
    
    plot_scree(our_pca, 
              save_path=os.path.join(results_dir, 'scree_plot.png'),
              max_components=30)
    
    plot_cumulative_variance(our_pca,
                            save_path=os.path.join(results_dir, 'cumulative_variance.png'))
    
    plot_pca_projection_2d(X, y, our_pca, sklearn_pca,
                          save_path=os.path.join(results_dir, 'pca_projection_2d.png'))
    
    plot_reconstruction_error(X, max_components=40,
                             save_path=os.path.join(results_dir, 'reconstruction_error.png'))
    
    plot_component_heatmap(our_pca, feature_names=feature_names, n_components=10,
                          save_path=os.path.join(results_dir, 'component_heatmap.png'))
    
    print("------ Все визуализации созданы! ------")
