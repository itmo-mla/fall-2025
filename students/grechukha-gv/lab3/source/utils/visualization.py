import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_svm_2d(svm, X, y, title="SVM Decision Boundary", save_path=None, samples=500):
    """
    Визуализирует разделяющую поверхность SVM в 2D с помощью PCA
    
    Args:
        svm: обученная модель SVM
        X: данные для визуализации (n, d)
        y: метки классов (n,) в формате {-1, +1}
        title: заголовок графика
        save_path: путь для сохранения
        samples: количество объектов для визуализации
    """
    
    if len(X) > samples:
        indices = np.random.choice(len(X), samples, replace=False)
        X_plot = X[indices]
        y_plot = y[indices]
    else:
        X_plot = X
        y_plot = y
    
    # Проецируем данные в 2D с помощью PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_plot)
    
    # Проецируем опорные векторы
    sv_pca = pca.transform(svm.support_vectors)
    
    # Создаем сетку для визуализации разделяющей поверхности
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Преобразуем сетку обратно в исходное пространство признаков
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_pca)
    
    Z = svm.decision_function(grid_original)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    
    contour = plt.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu')
    plt.colorbar(contour, label='Decision function value')
    
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    mask_neg = y_plot == -1
    mask_pos = y_plot == 1
    
    plt.scatter(X_pca[mask_neg, 0], X_pca[mask_neg, 1], 
                c='lightcoral', label='Класс -1', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    plt.scatter(X_pca[mask_pos, 0], X_pca[mask_pos, 1], 
                c='lightblue', label='Класс +1', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    
    # Выделяем опорные векторы
    plt.scatter(sv_pca[:, 0], sv_pca[:, 1], 
                s=200, facecolors='none', edgecolors='yellow', linewidths=3,
                label=f'Support Vectors ({len(sv_pca)})')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация сохранена в: {save_path}")
    
    plt.close()


def visualize_linear_svm_exact(svm, X, y, title="Linear SVM with Hyperplane", save_path=None):
    if X.shape[1] != 2:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Эта визуализация работает только для 2D данных (получено {X.shape[1]} признаков)")
        return
    
    if svm.kernel != 'linear':
        print("ПРЕДУПРЕЖДЕНИЕ: Эта визуализация работает только для линейного ядра")
        return
    
    plt.figure(figsize=(12, 9))
    
    mask_neg = y == -1
    mask_pos = y == 1
    
    plt.scatter(X[mask_neg, 0], X[mask_neg, 1], 
                c='red', label='Класс -1', alpha=0.6, s=60, edgecolors='k', linewidth=0.7)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], 
                c='blue', label='Класс +1', alpha=0.6, s=60, edgecolors='k', linewidth=0.7)
    
    # Выделяем опорные векторы
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                s=250, facecolors='none', edgecolors='yellow', linewidths=3,
                label=f'Support Vectors ({len(svm.support_vectors)})', zorder=10)
    
    # Для линейного ядра вычисляем w
    w = svm.get_weights()
    
    if w is not None:
        xlim = plt.xlim()
        x_vals = np.linspace(xlim[0], xlim[1], 100)
        
        # Разделяющая гиперплоскость: w·x + b = 0 => x2 = -(w1*x1 + b) / w2
        y_vals = -(w[0] * x_vals + svm.b) / w[1]
        plt.plot(x_vals, y_vals, 'g-', linewidth=3, label='Decision boundary (w·x + b = 0)', zorder=5)
        
        # Margin +1: w·x + b = 1
        y_vals_plus = -(w[0] * x_vals + svm.b - 1) / w[1]
        plt.plot(x_vals, y_vals_plus, 'g--', linewidth=2, alpha=0.7, label='Margin +1')
        
        # Margin -1: w·x + b = -1
        y_vals_minus = -(w[0] * x_vals + svm.b + 1) / w[1]
        plt.plot(x_vals, y_vals_minus, 'g--', linewidth=2, alpha=0.7, label='Margin -1')
        
        # Рисуем вектор нормали (направление w)
        # Берем центральную точку гиперплоскости
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = -(w[0] * x_center + svm.b) / w[1]
        
        # Нормализуем w для отображения
        w_norm = w / np.linalg.norm(w) * 2
        plt.arrow(x_center, y_center, w_norm[0], w_norm[1], 
                 head_width=0.3, head_length=0.2, fc='green', ec='darkgreen',
                 linewidth=2, label='Normal vector (w)', zorder=6)
    
    plt.xlabel('Feature 1', fontsize=13)
    plt.ylabel('Feature 2', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация линейного SVM сохранена в: {save_path}")
    
    plt.close()


def visualize_support_vectors(svm, X, y, title="Support Vectors Analysis", save_path=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    sv_pca = pca.transform(svm.support_vectors)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    mask_neg = y == -1
    mask_pos = y == 1
    
    ax.scatter(X_pca[mask_neg, 0], X_pca[mask_neg, 1], 
               c='lightcoral', label='Класс -1', alpha=0.4, s=30)
    ax.scatter(X_pca[mask_pos, 0], X_pca[mask_pos, 1], 
               c='lightblue', label='Класс +1', alpha=0.4, s=30)
    
    # Опорные векторы по классам
    sv_mask_neg = svm.support_vector_labels == -1
    sv_mask_pos = svm.support_vector_labels == 1
    
    ax.scatter(sv_pca[sv_mask_neg, 0], sv_pca[sv_mask_neg, 1], 
               s=150, marker='s', facecolors='red', edgecolors='black', linewidths=2,
               label=f'SV класс -1 ({np.sum(sv_mask_neg)})', alpha=0.8)
    ax.scatter(sv_pca[sv_mask_pos, 0], sv_pca[sv_mask_pos, 1], 
               s=150, marker='s', facecolors='blue', edgecolors='black', linewidths=2,
               label=f'SV класс +1 ({np.sum(sv_mask_pos)})', alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11)
    ax.set_title('Опорные векторы в 2D (PCA)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Правый график: гистограмма значений λ
    ax = axes[1]
    
    # Разделяем λ по классам
    lambdas_neg = svm.support_vector_lambdas[sv_mask_neg]
    lambdas_pos = svm.support_vector_lambdas[sv_mask_pos]
    
    ax.hist(lambdas_neg, bins=30, alpha=0.6, color='red', label=f'Класс -1', edgecolor='black')
    ax.hist(lambdas_pos, bins=30, alpha=0.6, color='blue', label=f'Класс +1', edgecolor='black')
    
    # Линия для C
    ax.axvline(x=svm.C, color='green', linestyle='--', linewidth=2, label=f'C = {svm.C}')
    
    ax.set_xlabel('Значение λ (множитель Лагранжа)', fontsize=11)
    ax.set_ylabel('Количество опорных векторов', fontsize=11)
    ax.set_title('Распределение значений λ', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация опорных векторов сохранена в: {save_path}")
    
    plt.close()


def visualize_with_tsne(X, y, svm=None, title="t-SNE Visualization", save_path=None, perplexity=30):
    print(f"Применение t-SNE (perplexity={perplexity})...")
    
    # Применяем t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    mask_neg = y == -1
    mask_pos = y == 1
    
    plt.scatter(X_tsne[mask_neg, 0], X_tsne[mask_neg, 1],
                c='red', label='Класс -1', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    plt.scatter(X_tsne[mask_pos, 0], X_tsne[mask_pos, 1],
                c='blue', label='Класс +1', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    
    if svm is not None and hasattr(svm, 'support_vector_indices'):
        sv_indices = svm.support_vector_indices
        plt.scatter(X_tsne[sv_indices, 0], X_tsne[sv_indices, 1],
                   s=200, facecolors='none', edgecolors='yellow',
                   linewidths=3, label=f'Support Vectors ({len(sv_indices)})')
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE визуализация сохранена в: {save_path}")
    
    plt.close()


def plot_kernel_comparison(results, save_path=None):
    kernels = [r['kernel'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    n_svs = [r['n_sv'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    bars = ax.bar(kernels, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Точность классификации по ядрам', fontsize=13, fontweight='bold')
    ax.set_ylim([min(accuracies) - 0.05, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax = axes[1]
    bars = ax.bar(kernels, n_svs, color=['skyblue', 'lightcoral', 'lightgreen'], 
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Количество опорных векторов', fontsize=12)
    ax.set_title('Количество опорных векторов по ядрам', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, n_sv in zip(bars, n_svs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{n_sv}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сравнение ядер сохранено в: {save_path}")
    
    plt.close()


def plot_pca_data(X, y, title="Dataset PCA Visualization", save_path=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    
    mask_neg = y == -1
    mask_pos = y == 1
    
    plt.scatter(X_pca[mask_neg, 0], X_pca[mask_neg, 1], 
                c='red', label='Класс -1 (не подписал)', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    plt.scatter(X_pca[mask_pos, 0], X_pca[mask_pos, 1], 
                c='blue', label='Класс +1 (подписал)', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    total_var = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.text(0.02, 0.98, f'Суммарная объясненная дисперсия: {total_var:.2%}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA визуализация сохранена в: {save_path}")
    
    plt.close()
