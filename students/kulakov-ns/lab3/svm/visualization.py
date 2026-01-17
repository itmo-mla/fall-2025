from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def plot_points_2d(
    X: Array,
    y: Array,
    title: str = "",
    support_mask: Array | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Simple 2D scatter plot for {-1,1} labels with optional support vectors."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="class +1", alpha=0.5)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label="class -1", alpha=0.5)

    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        ax.scatter(
            X[support_mask, 0],
            X[support_mask, 1],
            s=30,
            label="support vectors",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    return ax


def plot_decision_boundary_2d(
    model,
    X: Array,
    y: Array,
    title: str = "",
    support_mask: Array | None = None,
    grid_step: float = 0.02,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualize decision boundary for a fitted SVM-like model in 2D."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary_2d expects X with exactly 2 features")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.2)
    plot_points_2d(X, y, title=title, support_mask=support_mask, ax=ax)
    return ax
