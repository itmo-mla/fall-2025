from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_convergence_Q(histories: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    ensure_dir(out_path.parent)

    plt.figure()
    for name, h in histories.items():
        Q = np.asarray(h["Q"], dtype=float)
        it = np.arange(1, len(Q) + 1)
        plt.plot(it, Q, label=name)
    plt.title("Convergence of Q(w) = sum log(1 + exp(-y<w,x>))")
    plt.xlabel("Iteration")
    plt.ylabel("Q(w)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    plt.figure()
    for name, h in histories.items():
        s = np.asarray(h["step_norm"], dtype=float)
        it = np.arange(1, len(s) + 1)
        plt.plot(it, s, label=name)
    plt.title("Step norm convergence")
    plt.xlabel("Iteration")
    plt.ylabel("||w_{t+1} - w_t||")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_stepnorm.png"), dpi=180)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["-1", "+1"])
    plt.yticks([0, 1], ["-1", "+1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], out_path: Path) -> None:
    ensure_dir(out_path.parent)

    plt.figure()
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC curves (positive class = +1)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
