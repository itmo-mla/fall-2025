import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess_data(test_size: float = 0.3, random_state: int = 42) -> Tuple:
    # Загрузка данных
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Преобразование меток в {-1, +1}
    y = 2 * y - 1
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Размерность обучающей выборки: {X_train.shape}")
    print(f"Размерность тестовой выборки: {X_test.shape}")
    print(f"Баланс классов в train: {np.unique(y_train, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test


def visualize_margins(classifier, X: np.ndarray, y: np.ndarray, 
                     title: str = "Распределение отступов", 
                     save_path: str = None):

    margins = classifier.margin(X, y)
    
    plt.figure(figsize=(10, 6))
    
    # Гистограмма отступов
    plt.hist(margins, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Граница решения')
    plt.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Отступ = 1')
    plt.axvline(x=-1, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.xlabel('Отступ (M_i = y_i * (⟨w, x_i⟩ + b))')
    plt.ylabel('Количество объектов')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    predictions = classifier.predict(X)
    correct = margins > 0
    accuracy = np.mean(correct)
    
    plt.text(0.02, 0.95, f'Точность: {accuracy:.2%}', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.text(0.02, 0.88, f'Средний |M|: {np.mean(np.abs(margins)):.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_training_history(history: Dict, save_path: str = None):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # График потерь
    axes[0, 0].plot(history['loss'], linewidth=2)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Потери')
    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].grid(True, alpha=0.3)
    
    # График точности
    axes[0, 1].plot(history['accuracy'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Точность')
    axes[0, 1].set_title('Точность классификации')
    axes[0, 1].grid(True, alpha=0.3)
    
    # График среднего отступа
    axes[1, 0].plot(history['avg_margin'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Средний |M|')
    axes[1, 0].set_title('Средний модуль отступа')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График нормы весов
    axes[1, 1].plot(history['weights_norm'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('||w||')
    axes[1, 1].set_title('Норма вектора весов')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def multiclass_visualization(classifier, X: np.ndarray, y: np.ndarray, 
                           feature_indices: Tuple[int, int] = (0, 1),
                           title: str = "Разделяющая поверхность",
                           save_path: str = None):
    # Визуализация разделяющей поверхности в 2D.
    
    if X.shape[1] < 2:
        print("Недостаточно признаков для визуализации в 2D")
        return
    
    idx1, idx2 = feature_indices

    x_min, x_max = X[:, idx1].min() - 1, X[:, idx1].max() + 1
    y_min, y_max = X[:, idx2].min() - 1, X[:, idx2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid_data = np.zeros((xx.ravel().shape[0], X.shape[1]))
    grid_data[:, idx1] = xx.ravel()
    grid_data[:, idx2] = yy.ravel()

    Z = classifier.predict(grid_data)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)

    scatter = plt.scatter(X[:, idx1], X[:, idx2], c=y, 
                         cmap=plt.cm.coolwarm, edgecolors='k', s=50)
    
    plt.xlabel(f'Признак {idx1}')
    plt.ylabel(f'Признак {idx2}')
    plt.title(title)
    plt.colorbar(scatter, label='Класс')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
