import io
import zipfile

import numpy as np
import pandas as pd
import requests
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_preprocess_data():

    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    print("Загрузка Bank Marketing Dataset...")
    response = requests.get(url)

    with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
        with outer_zip.open('bank-additional.zip') as inner_zip_file:
            inner_zip_data = inner_zip_file.read()
            with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:
                with inner_zip.open('bank-additional/bank-additional-full.csv') as f:
                    bank_marketing = pd.read_csv(f, sep=';')

    print(f"Датасет загружен: {len(bank_marketing)} объектов")
    
    # Преобразуем целевую переменную: yes -> 1, no -> 0
    bank_marketing['y'] = bank_marketing['y'].map({'yes': 1, 'no': 0}).astype(int)
    
    # Удаляем duration (не должен использоваться для предсказания)
    bank_marketing = bank_marketing.drop('duration', axis=1)

    X = bank_marketing.drop('y', axis=1)
    y = bank_marketing['y']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    numerical_columns = [
        'age', 'campaign', 'pdays', 'previous', 
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
        'euribor3m', 'nr.employed'
    ]
    categorical_columns = [
        'job', 'marital', 'education', 'default', 'housing', 
        'loan', 'contact', 'month', 'day_of_week', 'poutcome'
    ]

    # Преобразуем категориальные признаки
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_train_encoded_df = pd.DataFrame(
        X_train_encoded, columns=encoded_columns, index=X_train.index
    )
    X_test_encoded_df = pd.DataFrame(
        X_test_encoded, columns=encoded_columns, index=X_test.index
    )

    X_train_final = pd.concat([X_train[numerical_columns], X_train_encoded_df], axis=1)
    X_test_final = pd.concat([X_test[numerical_columns], X_test_encoded_df], axis=1)

    # Нормализуем численные признаки
    scaler = StandardScaler()
    X_train_final[numerical_columns] = scaler.fit_transform(X_train_final[numerical_columns])
    X_test_final[numerical_columns] = scaler.transform(X_test_final[numerical_columns])

    print(f"Обучающая выборка: {len(X_train_final)} объектов")
    print(f"Тестовая выборка: {len(X_test_final)} объектов")
    print(f"Количество признаков: {X_train_final.shape[1]}")
    print(f"Распределение классов в train: {np.bincount(y_train)}")
    print(f"Распределение классов в test: {np.bincount(y_test)}")

    return X_train_final.values, X_test_final.values, y_train.values, y_test.values


def visualize_data_tsne(X, y, title="t-SNE визуализация датасета", save_path=None, perplexity=30):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    print(f"Выполняется t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    
    mask_neg = y == 0
    mask_pos = y == 1
    
    plt.scatter(
        X_tsne[mask_neg, 0], X_tsne[mask_neg, 1],
        c='red', label='Класс 0 (не подписал)',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.5
    )
    plt.scatter(
        X_tsne[mask_pos, 0], X_tsne[mask_pos, 1],
        c='blue', label='Класс 1 (подписал)',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.5
    )
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.text(
        0.02, 0.98, f'Perplexity: {perplexity}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE визуализация сохранена в: {save_path}")
    
    plt.close()
    
    return X_tsne


def visualize_data_pca(X, y, title="PCA визуализация датасета", save_path=None):

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Применяем PCA для снижения размерности до 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    
    mask_neg = y == 0
    mask_pos = y == 1
    
    plt.scatter(
        X_pca[mask_neg, 0], X_pca[mask_neg, 1], 
        c='red', label='Класс 0 (не подписал)', 
        alpha=0.6, s=30, edgecolors='k', linewidth=0.5
    )
    plt.scatter(
        X_pca[mask_pos, 0], X_pca[mask_pos, 1], 
        c='blue', label='Класс 1 (подписал)', 
        alpha=0.6, s=30, edgecolors='k', linewidth=0.5
    )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    total_var = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    plt.text(
        0.02, 0.98, f'Суммарная объясненная дисперсия: {total_var:.2%}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA визуализация сохранена в: {save_path}")
    
    plt.close()
    
    return X_pca, pca


def apply_smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Применяет SMOTE (Synthetic Minority Over-sampling Technique) для балансировки классов.
    
    Args:
        X_train: обучающая выборка признаков
        y_train: метки классов
        sampling_strategy: стратегия сэмплирования ('auto', float, или dict)
                          'auto' - делает классы равными
                          float - соотношение minority/majority после балансировки
                          dict - конкретное количество объектов для каждого класса
        k_neighbors: количество соседей для генерации синтетических объектов
        random_state: random seed для воспроизводимости
    
    Returns:
        X_resampled, y_resampled: сбалансированные данные
    """
    print(f"\nПрименение SMOTE для балансировки классов...")
    print(f"До SMOTE: {np.bincount(y_train)}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"После SMOTE: {np.bincount(y_resampled)}")
    print(f"Добавлено {len(y_resampled) - len(y_train)} синтетических объектов класса 1\n")
    
    return X_resampled, y_resampled


X_train_array, X_test_array, y_train_array, y_test_array = load_and_preprocess_data()
