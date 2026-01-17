import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def save_fig(save_path: Optional[str] = None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    if 'recurrent_loss' in history and len(history['recurrent_loss']) > 0:
        n_samples_per_epoch = len(history['recurrent_loss']) // len(history['train_loss']) if len(history['train_loss']) > 0 else 1
        recurrent_epochs = [history['recurrent_loss'][i * n_samples_per_epoch - 1] 
                           for i in range(1, len(history['train_loss']) + 1) 
                           if i * n_samples_per_epoch <= len(history['recurrent_loss'])]
        if len(recurrent_epochs) > 0:
            axes[0, 0].plot(range(len(recurrent_epochs)), recurrent_epochs, 
                          label='Recurrent Loss', linewidth=2, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history and len(history['val_acc']) > 0:
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'margins' in history and len(history['margins']) > 0:
        margins = history['margins'][-1]
        axes[1, 0].hist(margins, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Decision Boundary')
        axes[1, 0].set_xlabel('Margin')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Margins (Last Epoch)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'margins' in history and len(history['margins']) > 0:
        mean_margins = [np.mean(m) for m in history['margins']]
        std_margins = [np.std(m) for m in history['margins']]
        epochs = range(len(mean_margins))
        axes[1, 1].plot(epochs, mean_margins, label='Mean Margin', linewidth=2)
        axes[1, 1].fill_between(epochs, 
                               np.array(mean_margins) - np.array(std_margins),
                               np.array(mean_margins) + np.array(std_margins),
                               alpha=0.3, label='±1 Std')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Decision Boundary')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Margin')
        axes[1, 1].set_title('Evolution of Margins')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_fig(save_path)
    
    plt.close()


def plot_margins_analysis(model, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
    margins = model.margin(X, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(margins[y == 1], bins=30, alpha=0.7, label='Class +1', color='green', edgecolor='black')
    axes[0].hist(margins[y == -1], bins=30, alpha=0.7, label='Class -1', color='red', edgecolor='black')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    axes[0].set_xlabel('Margin')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Margins by Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    data_to_plot = [margins[y == 1], margins[y == -1]]
    bp = axes[1].boxplot(data_to_plot, labels=['Class +1', 'Class -1'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    axes[1].set_ylabel('Margin')
    axes[1].set_title('Box Plot of Margins by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_fig(save_path)
    
    plt.close()


def plot_comparison_histogram(results: Dict[str, float], save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(results.keys())
    accuracies = list(results.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Comparison of Different Methods and SGDClassifier', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(accuracies) * 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    save_fig(save_path)
    
    plt.close()


def plot_weights_visualization(model, feature_names: Optional[List[str]] = None, 
                               save_path: Optional[str] = None):
    w, b = model.get_weights()
    
    fig, ax = plt.subplots(figsize=(12, 6))

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(w))]

    colors = ['green' if x > 0 else 'red' for x in w]
    ax.barh(feature_names, w, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Model Weights (Bias: {b:.4f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    save_fig(save_path)
    
    plt.close()

