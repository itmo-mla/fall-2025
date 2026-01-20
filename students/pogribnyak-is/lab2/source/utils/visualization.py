import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Visualizer:
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

    @staticmethod
    def _save_and_show(save_path: Optional[str] = None):
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_empirical_risk(self, k_range: List[int], errors: List[float], save_path: Optional[str] = None):
        plt.figure(figsize=self.figsize)
        plt.plot(k_range, errors, marker='o', linewidth=2, markersize=8)
        plt.xlabel('k (количество соседей)', fontsize=12)
        plt.ylabel('Эмпирический риск (LOO)', fontsize=12)
        plt.title('Зависимость эмпирического риска от параметра k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        best_idx = np.argmin(errors)
        plt.axvline(x=k_range[best_idx], color='r', linestyle='--', alpha=0.7, label=f'Лучшее k={k_range[best_idx]}')
        plt.scatter([k_range[best_idx]], [errors[best_idx]], color='r', s=200, zorder=5)
        plt.legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prototype_selection(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_prototypes: np.ndarray,
        y_prototypes: np.ndarray,
        save_path: Optional[str] = None,
    ):
        reducer = PCA(n_components=2, random_state=42)

        X_2d_original = reducer.fit_transform(X_original)
        X_2d_prototypes = reducer.transform(X_prototypes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        unique_classes = np.unique(y_original)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            mask = y_original == cls
            ax1.scatter(
                X_2d_original[mask, 0],
                X_2d_original[mask, 1],
                c=[colors[i]],
                label=f'Класс {cls}',
                alpha=0.6,
                s=50
            )
        
        ax1.set_title('Исходная выборка', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PCA компонента 1')
        ax1.set_ylabel(f'PCA компонента 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        for i, cls in enumerate(unique_classes):
            mask = y_prototypes == cls
            ax2.scatter(
                X_2d_prototypes[mask, 0],
                X_2d_prototypes[mask, 1],
                c=[colors[i]],
                label=f'Класс {cls}',
                alpha=0.8,
                s=100,
                edgecolors='black',
                linewidth=2
            )
        
        ax2.set_title(f'Отобранные эталоны (n={len(X_prototypes)})', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PCA компонента 1')
        ax2.set_ylabel(f'PCA компонента 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self._save_and_show(save_path)

    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Матрица ошибок',
        save_path: Optional[str] = None
    ):
        from utils.metrics import ClassificationMetrics
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Количество примеров'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        self._save_and_show(save_path)
    
    def plot_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        algorithms = list(results.keys())
        
        scalar_metrics = ['accuracy', 'precision', 'recall', 'f1']
        available_metrics = []
        
        for metric in scalar_metrics:
            if metric in results[algorithms[0]]:
                value = results[algorithms[0]][metric]
                try:
                    float(value)
                    available_metrics.append(metric)
                except (TypeError, ValueError):
                    continue
        
        if not available_metrics:
            print("Предупреждение: нет доступных скалярных метрик для сравнения")
            return
        
        x = np.arange(len(available_metrics))
        width = 0.8 / len(algorithms)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, alg in enumerate(algorithms):
            values = [float(results[alg][m]) for m in available_metrics]
            offset = (i - len(algorithms) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=alg, alpha=0.8)
        
        ax.set_xlabel('Метрики', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.set_title('Сравнение алгоритмов', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(available_metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        self._save_and_show(save_path)

