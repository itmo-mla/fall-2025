from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def show_pairs(df: pd.DataFrame, save_path: Path) -> None:
    sns.set(style="ticks", context="notebook")
    g = sns.pairplot(df, hue="LeaveOrNot", diag_kind="kde")
    g.fig.suptitle("Pairplot по признакам (LeaveOrNot)", y=1.02)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, bbox_inches="tight")
    plt.close(g.fig)


def show_dist(df: pd.DataFrame, save_path: Path) -> None:
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="LeaveOrNot", data=df, ax=ax)
    ax.set_title("Распределение целевой переменной LeaveOrNot")
    ax.set_xlabel("LeaveOrNot")
    ax.set_ylabel("Count")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compare_results(
    my_scores: Dict[str, float],
    sklearn_scores: Dict[str, float],
    save_path: Path,
) -> None:
    sns.set(style="whitegrid")
    kernels = list(my_scores.keys())
    my_vals = [my_scores[k] for k in kernels]
    sklearn_vals = [sklearn_scores[k] for k in kernels]

    x = range(len(kernels))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35

    ax.bar([i - width / 2 for i in x], my_vals, width=width, label="Custom SVM")
    ax.bar([i + width / 2 for i in x], sklearn_vals, width=width, label="sklearn SVC")

    ax.set_xticks(list(x))
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Custom SVM vs sklearn SVC")
    ax.legend()

    for i, v in enumerate(my_vals):
        ax.text(i - width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(sklearn_vals):
        ax.text(i + width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def visualize_model(
    model,
    X,
    y,
    save_path: Path,
    kernel_name: str = "",
) -> None:
    X = np.asarray(X)
    y = np.asarray(y)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    
    grid_original = pca.inverse_transform(grid_2d)
    Z_pred = model.predict(grid_original).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.contour(xx, yy, Z_pred, levels=[0.5], colors='black', linewidths=2)
    
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap=plt.cm.RdYlBu,
        alpha=0.6,
        s=30,
    )
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title = f"Разделяющая граница"
    if kernel_name:
        title += f" ({kernel_name})"
    ax.set_title(title)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path: Path,
    kernel_name: str = "",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Количество'})
    
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    title = 'Confusion Matrix'
    if kernel_name:
        title += f' ({kernel_name})'
    ax.set_title(title)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



