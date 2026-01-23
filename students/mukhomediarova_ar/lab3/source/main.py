from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    # Запуск как пакет: python -m students.mukhomediarova_ar.lab3.source.main
    from .data_utils import (
        RNG_SEED,
        load_dataset,
        standardize_features,
        train_val_test_split,
    )
    from .svm import KernelSVM, accuracy_score, linear_kernel, make_rbf_kernel
except ImportError:  # pragma: no cover - запуск напрямую как скрипта
    from data_utils import (  # type: ignore
        RNG_SEED,
        load_dataset,
        standardize_features,
        train_val_test_split,
    )
    from svm import KernelSVM, accuracy_score, linear_kernel, make_rbf_kernel  # type: ignore


try:
    # Эталонная реализация SVM из sklearn для сравнения.
    # При проверке допускается отсутствие sklearn; в таком случае сравнение пропускается.
    from sklearn.svm import SVC
except Exception:  # pragma: no cover - sklearn может быть не установлен
    SVC = None  # type: ignore


def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_decision_boundary(
    model,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    filename: str,
    support_mask: np.ndarray | None = None,
) -> None:
    """Визуализация границы решения SVM для двумерного датасета."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = x[:, 0].min() - 1.0, x[:, 0].max() + 1.0
    y_min, y_max = x[:, 1].min() - 1.0, x[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, "decision_function"):
        zz = model.decision_function(grid)
    else:
        zz = model.predict(grid)  # type: ignore[assignment]
    zz = zz.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    # Фон по значению решающей функции
    plt.contourf(
        xx,
        yy,
        zz,
        levels=30,
        cmap="coolwarm",
        alpha=0.4,
    )

    # Линия раздела f(x) = 0 и, по возможности, полосы отступа
    plt.contour(
        xx,
        yy,
        zz,
        levels=[-1.0, 0.0, 1.0],
        colors=["blue", "black", "red"],
        linestyles=["--", "-", "--"],
        linewidths=1.2,
    )

    # Обучающие точки
    scatter = plt.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        s=30,
        alpha=0.9,
        label="Обучающие объекты",
    )

    # Опорные векторы (если передана маска)
    if support_mask is not None and np.any(support_mask):
        plt.scatter(
            x[support_mask, 0],
            x[support_mask, 1],
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.5,
            s=80,
            label="Опорные векторы",
        )

    plt.xlabel("Признак 1 (стандартизованный)")
    plt.ylabel("Признак 2 (стандартизованный)")
    plt.title(title)
    plt.colorbar(scatter, label="Класс")
    plt.legend(loc="best")

    out_path = _results_dir() / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Граница решения сохранена в {out_path}")


def run_experiments() -> None:
    # 1. Датасет
    x, y = load_dataset()
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = train_val_test_split(x, y)

    x_train, x_val, x_test = standardize_features(x_train, x_val, x_test)

    print(
        f"Размерности выборок: train={x_train.shape[0]}, "
        f"val={x_val.shape[0]}, test={x_test.shape[0]}"
    )

    # 2. Линейный SVM (линейное ядро)
    print("\n=== Линейный SVM (линейное ядро) ===")
    svm_linear = KernelSVM(c=1.0, kernel=linear_kernel, tol=1e-5, max_iter=300)
    svm_linear.fit(x_train, y_train)

    y_val_pred_lin = svm_linear.predict(x_val)
    y_test_pred_lin = svm_linear.predict(x_test)
    val_acc_lin = accuracy_score(y_val, y_val_pred_lin)
    test_acc_lin = accuracy_score(y_test, y_test_pred_lin)
    print(f"Линейный SVM: val_accuracy={val_acc_lin:.4f}, test_accuracy={test_acc_lin:.4f}")

    support_mask_train = np.zeros(x_train.shape[0], dtype=bool)
    if svm_linear.support_indices_ is not None:
        support_mask_train[svm_linear.support_indices_] = True

    _plot_decision_boundary(
        svm_linear,
        x_train,
        y_train,
        title="Линейный SVM (kernel = linear)",
        filename="svm_linear_decision_boundary.png",
        support_mask=support_mask_train,
    )

    # 3. Ядровой SVM с RBF‑ядром (kernel trick)
    print("\n=== Ядровой SVM (RBF‑ядро) ===")
    rbf_kernel = make_rbf_kernel(gamma=1.0)
    svm_rbf = KernelSVM(c=1.0, kernel=rbf_kernel, tol=1e-5, max_iter=300)
    svm_rbf.fit(x_train, y_train)

    y_val_pred_rbf = svm_rbf.predict(x_val)
    y_test_pred_rbf = svm_rbf.predict(x_test)
    val_acc_rbf = accuracy_score(y_val, y_val_pred_rbf)
    test_acc_rbf = accuracy_score(y_test, y_test_pred_rbf)
    print(f"RBF‑SVM: val_accuracy={val_acc_rbf:.4f}, test_accuracy={test_acc_rbf:.4f}")

    support_mask_train_rbf = np.zeros(x_train.shape[0], dtype=bool)
    if svm_rbf.support_indices_ is not None:
        support_mask_train_rbf[svm_rbf.support_indices_] = True

    _plot_decision_boundary(
        svm_rbf,
        x_train,
        y_train,
        title="Ядровой SVM (kernel = RBF)",
        filename="svm_rbf_decision_boundary.png",
        support_mask=support_mask_train_rbf,
    )

    # 4. Сравнение с эталонной реализацией из sklearn.svm.SVC (если доступна)
    if SVC is not None:
        print("\n=== Эталонная реализация SVM из sklearn.svm.SVC ===")
        svc = SVC(
            kernel="rbf",
            C=1.0,
            gamma=1.0,
        )
        svc.fit(x_train, (y_train > 0).astype(int))

        y_val_pred_svc = svc.predict(x_val)
        y_test_pred_svc = svc.predict(x_test)

        # Переводим предсказания {0, 1} в {-1, 1} для корректного сравнения.
        y_val_pred_svc_pm = np.where(y_val_pred_svc == 1, 1.0, -1.0)
        y_test_pred_svc_pm = np.where(y_test_pred_svc == 1, 1.0, -1.0)

        val_acc_svc = accuracy_score(y_val, y_val_pred_svc_pm)
        test_acc_svc = accuracy_score(y_test, y_test_pred_svc_pm)

        print(
            "Сравнение по accuracy (val / test):\n"
            f"  Наш линейный SVM:  {val_acc_lin:.4f} / {test_acc_lin:.4f}\n"
            f"  Наш RBF‑SVM:       {val_acc_rbf:.4f} / {test_acc_rbf:.4f}\n"
            f"  sklearn SVC (RBF): {val_acc_svc:.4f} / {test_acc_svc:.4f}"
        )

        # Визуализируем решение SVC для наглядного сравнения
        _plot_decision_boundary(
            svc,
            x_train,
            (y_train > 0).astype(int),
            title="Эталонный SVC (kernel = RBF)",
            filename="svm_sklearn_rbf_decision_boundary.png",
            support_mask=None,
        )
    else:
        print(
            "\nsklearn.svm.SVC недоступен, сравнение с эталонной реализацией "
            "SVM из sklearn пропущено."
        )


def main() -> None:
    np.random.seed(RNG_SEED)
    run_experiments()


if __name__ == "__main__":
    main()

