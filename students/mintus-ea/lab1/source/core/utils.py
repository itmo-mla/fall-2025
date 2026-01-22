import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_history(history, title="Training History", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_margins(margins, title="Margin Distribution", save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(margins, kde=True, bins=30)
    plt.title(title)
    plt.xlabel('Margin')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()
