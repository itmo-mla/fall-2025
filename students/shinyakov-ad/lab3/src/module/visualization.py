from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pairplot(df: pd.DataFrame, save_path: Path) -> None:
    sns.set(style="ticks", context="notebook")
    g = sns.pairplot(df, hue="LeaveOrNot", diag_kind="kde")
    g.fig.suptitle("Pairplot по признакам (LeaveOrNot)", y=1.02)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, bbox_inches="tight")
    plt.close(g.fig)


def plot_target_distribution(df: pd.DataFrame, save_path: Path) -> None:
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="LeaveOrNot", data=df, ax=ax)
    ax.set_title("Распределение целевой переменной LeaveOrNot")
    ax.set_xlabel("LeaveOrNot")
    ax.set_ylabel("Count")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(
    custom_scores: Dict[str, float],
    baseline_scores: Dict[str, float],
    save_path: Path,
) -> None:
    sns.set(style="whitegrid")
    kernels = list(custom_scores.keys())
    custom_vals = [custom_scores[k] for k in kernels]
    baseline_vals = [baseline_scores[k] for k in kernels]

    x = range(len(kernels))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35

    ax.bar([i - width / 2 for i in x], custom_vals, width=width, label="Custom SVM")
    ax.bar([i + width / 2 for i in x], baseline_vals, width=width, label="sklearn SVC")

    ax.set_xticks(list(x))
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Сравнение accuracy: custom SVM vs sklearn SVC")
    ax.legend()

    for i, v in enumerate(custom_vals):
        ax.text(i - width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(baseline_vals):
        ax.text(i + width / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



