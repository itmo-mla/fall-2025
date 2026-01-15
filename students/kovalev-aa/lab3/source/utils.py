import pandas as pd
import numpy as np 
import os
from sklearn.impute import SimpleImputer
import urllib
import urllib.request

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_base():
    file_path = "breast-cancer.csv" 
    if not os.path.exists(file_path):
        print(f"{file_path} не найден. Скачиваем...")
        url = (
          "https://huggingface.co/datasets/mnemoraorg/wisconsin-breast-cancer-diagnostic/"
          "raw/main/raw_breast_cancer.csv"
        )
        urllib.request.urlretrieve(url, file_path)
        print(f"Файл скачан и сохранён в {file_path}")

    clean_data = pd.read_csv(file_path)
    clean_data = clean_data.drop(columns=['id'])
    clean_data['diagnosis'] = clean_data['diagnosis'].map({'M': 1, 'B': -1})

    x = clean_data.drop('diagnosis', axis=1).to_numpy(dtype=float)
    y = clean_data['diagnosis'].to_numpy(dtype=int)

    # Заполняем пропуски средними значениями по колонкам
    imputer = SimpleImputer(strategy="mean")
    x = imputer.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
# def load_base():
#     # Загружаем данные
#     file_path = "Iris.csv"
#     df = pd.read_csv(file_path)
    
#     # Убираем колонку Id
#     df = df.drop(columns=['Id'])
    
#     # Выбираем только два класса: setosa и versicolor
#     df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy()
    
#     # Преобразуем species в -1 и +1
#     df['Species'] = df['Species'].map({
#         'Iris-setosa': -1,
#         'Iris-versicolor': 1
#     })
    
#     # Признаки и метки
#     X = df.drop('Species', axis=1).to_numpy(dtype=float)
#     y = df['Species'].to_numpy(dtype=int)
    
#     # Разделение на тренировочную и тестовую выборки
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )
    
#     return X_train, X_test, y_train, y_test
# def load_base():
#     # Путь к файлу Mushrooms.csv
#     file_path = "Mushrooms.csv"
#     df = pd.read_csv(file_path)
    
#     # Целевая переменная
#     y = df['Poisonous'].map({0: -1, 1: 1}).sample(frac=0.1).to_numpy(dtype=int)  # -1 = Edible, +1 = Poisonous
    
#     # Признаки
#     X = df.drop(columns=['Poisonous']).sample(frac=0.1).to_numpy(dtype=float)
    
    
#     # Разделение на тренировочную и тестовую выборки
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )
    
#     return X_train, X_test, y_train, y_test

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def visualize_t_sne_and_pca(x, y, support_vectors=None, tsne_save_path="tsne_2d.png", pca_save_path="pca_2d.png"):
    # ------------------ t-SNE ------------------
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    
    if support_vectors is not None:
        combined = np.vstack([x, support_vectors])
        X_tsne_combined = tsne.fit_transform(combined)
        X_tsne = X_tsne_combined[:len(x)]
        X_sv_tsne = X_tsne_combined[len(x):]
    else:
        X_tsne = tsne.fit_transform(x)
        X_sv_tsne = None

    plt.figure(figsize=(8,6))
    for class_id in np.unique(y):
        plt.scatter(X_tsne[y==class_id, 0], X_tsne[y==class_id, 1], label=f"Class {class_id}", alpha=0.7)
    if X_sv_tsne is not None:
        plt.scatter(X_sv_tsne[:, 0], X_sv_tsne[:, 1], color='red', marker='*', s=150, label='Support Vectors')
    plt.legend()
    plt.title("t-SNE 2D visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ------------------ PCA ------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    
    plt.figure(figsize=(8,6))
    for class_id in np.unique(y):
        plt.scatter(X_pca[y==class_id, 0], X_pca[y==class_id, 1], label=f"Class {class_id}", alpha=0.7)
    
    if support_vectors is not None:
        X_sv_pca = pca.transform(support_vectors)
        plt.scatter(X_sv_pca[:, 0], X_sv_pca[:, 1], color='red', marker='*', s=150, label='Support Vectors')
    
    plt.legend()
    plt.title("PCA 2D visualization")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(pca_save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_t_sne_3d(x, y, support_vectors=None, save_path="tsne_3d.png"):
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(x)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Основные классы
    for class_id in np.unique(y):
        ax.scatter(
            X_tsne[y == class_id, 0],
            X_tsne[y == class_id, 1],
            X_tsne[y == class_id, 2],
            label=f"Class {class_id}",
            s=50,
            alpha=0.8
        )

    # Опорные векторы, если переданы
    if support_vectors is not None:
        sv_tsne = tsne.fit_transform(support_vectors)
        ax.scatter(
            sv_tsne[:, 0],
            sv_tsne[:, 1],
            sv_tsne[:, 2],
            color='red',
            marker='*',
            s=150,
            label='Support Vectors'
        )

    ax.set_title("t-SNE 3D visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.legend()

    # Сохранение графика
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_risk_summary(k_array, k_errors, save_path="risk_plots"): 
    
    mean_errors = np.mean(k_errors, axis=1)
    n_samples = k_errors.shape[1]
    
    # Определяем индексы для лучшего, худшего и среднего k
    best_idx = np.argmin(mean_errors)
    worst_idx = np.argmax(mean_errors)
    median_val = np.median(mean_errors)
    mid_idx = np.argmin(np.abs(mean_errors - median_val))
    
    best_k = k_array[best_idx]
    worst_k = k_array[worst_idx]
    mid_k = k_array[mid_idx]
    
    # 1) График ошибок по среднему по k
    plt.figure(figsize=(10, 6))
    plt.plot(k_array, mean_errors, marker='o', linewidth=2, markersize=6)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Лучшее k={best_k}')
    plt.axvline(x=worst_k, color='orange', linestyle='--', alpha=0.7, label=f'Худшее k={worst_k}')
    plt.axvline(x=mid_k, color='green', linestyle='--', alpha=0.7, label=f'Среднее k={mid_k}')
    plt.xlabel('k')
    plt.ylabel('Средняя ошибка (LOO)')
    plt.title('Средняя ошибка для разных значений k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_mean_errors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2) График ошибок по номеру исключенного элемента для худшего k
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_samples + 1), k_errors[worst_idx, :], 
             marker='o', color='red', linewidth=1, markersize=3)
    plt.axhline(y=np.mean(k_errors[worst_idx, :]), color='darkred', 
                linestyle='--', label=f'Среднее = {np.mean(k_errors[worst_idx, :]):.3f}')
    plt.xlabel('Номер исключенного элемента')
    plt.ylabel('Ошибка LOO')
    plt.title(f'Ошибки LOO для худшего k={worst_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_worst_k_errors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3) График ошибок по номеру исключенного элемента для лучшего k
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_samples + 1), k_errors[best_idx, :], 
             marker='o', color='green', linewidth=1, markersize=3)
    plt.axhline(y=np.mean(k_errors[best_idx, :]), color='darkgreen', 
                linestyle='--', label=f'Среднее = {np.mean(k_errors[best_idx, :]):.3f}')
    plt.xlabel('Номер исключенного элемента')
    plt.ylabel('Ошибка LOO')
    plt.title(f'Ошибки LOO для лучшего k={best_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_best_k_errors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4) График ошибок по номеру исключенного элемента для среднего k
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_samples + 1), k_errors[mid_idx, :], 
             marker='o', color='blue', linewidth=1, markersize=3)
    plt.axhline(y=np.mean(k_errors[mid_idx, :]), color='darkblue', 
                linestyle='--', label=f'Среднее = {np.mean(k_errors[mid_idx, :]):.3f}')
    plt.xlabel('Номер исключенного элемента')
    plt.ylabel('Ошибка LOO')
    plt.title(f'Ошибки LOO для среднего k={mid_k}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_mid_k_errors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены как:")
    print(f"1. {save_path}_mean_errors.png - средние ошибки по k")
    print(f"2. {save_path}_worst_k_errors.png - ошибки для худшего k={worst_k}")
    print(f"3. {save_path}_best_k_errors.png - ошибки для лучшего k={best_k}")
    print(f"4. {save_path}_mid_k_errors.png - ошибки для среднего k={mid_k}")


def print_metrics(metrics_dict):
    """Красивый вывод метрик"""
    print("\n" + "="*50)
    print("МЕТРИКИ КЛАССИФИКАЦИИ")
    print("="*50)
    
    print(f"Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall:    {metrics_dict['recall']:.4f}")
    print(f"F1-score:  {metrics_dict['f1']:.4f}")
    
 
    print("\nМатрица ошибок:")
    print(metrics_dict['confusion_matrix'])
    print("="*50)
# def load_base():
#         # Set the path to the file you'd like to load 

#     file_path = "Iris.csv"

#     # Load the latest version
#     clean_data = pd.read_csv(file_path) 
#     clean_data = clean_data.drop(columns=['Id'])
#     clean_data['Species'] = pd.Categorical(clean_data['Species']).codes


#     y = np.array(clean_data['Species'])
#     x = np.array(clean_data.drop(columns=['Species']))

#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#     return x_train,x_test,y_train,y_test

 