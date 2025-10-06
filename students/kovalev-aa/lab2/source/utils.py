import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

class ModelWrapper:
    def __init__(self, model, test_accuracy=None,
                 precision=None, recall=None, f1=None):
        self.model = model
        self.test_accuracy = test_accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
import pandas as pd
import numpy as np
from datetime import datetime

def load_base():
    # Путь к файлу
    file_path = "breast-cancer.csv" 
    clean_data = pd.read_csv(file_path)
    
 
    clean_data = clean_data.drop(columns=['id'])
    
 
    clean_data['diagnosis'] = clean_data['diagnosis'].map({'M': 1, 'B': 0})
    
  
    
 
    train_x = clean_data.drop('diagnosis', axis=1).to_numpy(dtype=float)
    train_y = clean_data['diagnosis'].to_numpy(dtype=int) 
    
    return train_x, train_y 

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

 