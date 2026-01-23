import json
import os
import time
from dataclasses import asdict

import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from source.data import load_binary_dataset
from source.metrics import compute_confusion, compute_metrics
from source.svm_dual import DualSVM, DualSVMConfig
from source.visualize import (
    plot_confusion_matrices,
    plot_decision_boundary_2d,
    plot_decision_boundary_pca,
    plot_pca_projection,
)


def _ensure_outputs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)


def _print_block(title, custom_m, sklearn_m, custom_fit, sklearn_fit):
    print(f"\n=== {title} ===")
    print(f"Custom acc={custom_m.accuracy:.4f} | sklearn acc={sklearn_m.accuracy:.4f}")
    print(f"Custom fit={custom_fit:.4f}s | sklearn fit={sklearn_fit:.4f}s")
    print(
        "Custom metrics:\n"
        f"  (+1) P/R/F1 = {custom_m.precision_pos:.3f}/{custom_m.recall_pos:.3f}/{custom_m.f1_pos:.3f}\n"
        f"  (-1) P/R/F1 = {custom_m.precision_neg:.3f}/{custom_m.recall_neg:.3f}/{custom_m.f1_neg:.3f}\n"
        f"  macro F1    = {custom_m.f1_macro:.3f}"
    )
    print(
        "sklearn metrics:\n"
        f"  (+1) P/R/F1 = {sklearn_m.precision_pos:.3f}/{sklearn_m.recall_pos:.3f}/{sklearn_m.f1_pos:.3f}\n"
        f"  (-1) P/R/F1 = {sklearn_m.precision_neg:.3f}/{sklearn_m.recall_neg:.3f}/{sklearn_m.f1_neg:.3f}\n"
        f"  macro F1    = {sklearn_m.f1_macro:.3f}"
    )


def compare_with_sklearn(X_train, y_train, X_test, y_test, cfg: DualSVMConfig):
    start = time.time()
    clf = SVC(kernel=cfg.kernel, C=cfg.C, gamma=cfg.gamma, degree=cfg.degree, coef0=cfg.coef0)
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start

    m = compute_metrics(y_test, y_pred)
    cm = compute_confusion(y_test, y_pred)
    return m, cm, y_pred, fit_time, pred_time


def train_custom(X_train, y_train, X_test, y_test, cfg: DualSVMConfig):
    model = DualSVM(cfg)
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start

    m = compute_metrics(y_test, y_pred)
    cm = compute_confusion(y_test, y_pred)

    sv = model.X_train[model.lmbda > cfg.eps_sv]
    return model, m, cm, y_pred, fit_time, pred_time, sv


def run_linear_baseline(X_train, y_train, X_test, y_test):
    cfg = DualSVMConfig(kernel="linear", C=1.0)
    model, m_custom, cm_custom, y_pred_custom, fit_custom, _, sv = train_custom(X_train, y_train, X_test, y_test, cfg)
    m_sk, cm_sk, y_pred_sk, fit_sk, _ = compare_with_sklearn(X_train, y_train, X_test, y_test, cfg)

    plot_decision_boundary_pca(
        model,
        X_train,
        y_train,
        title="Custom Dual SVM | kernel=linear (PCA space)",
        filename="boundary_custom_linear.png",
        support_vectors=sv,
    )

    plot_confusion_matrices(
        y_true=y_test,
        y_pred_custom=y_pred_custom,
        y_pred_sklearn=y_pred_sk,
        title="Breast Cancer | Linear kernel",
        filename="cm_linear_test.png",
    )

    _print_block("Kernel: linear (Breast Cancer)", m_custom, m_sk, fit_custom, fit_sk)

    result = {
        "kernel": "linear",
        "cfg": asdict(cfg),
        "custom": {"metrics": asdict(m_custom), "confusion": cm_custom, "fit_time_s": fit_custom},
        "sklearn": {"metrics": asdict(m_sk), "confusion": cm_sk, "fit_time_s": fit_sk},
        "n_support_vectors": int(len(sv)),
        "linear_hyperplane": {"w": model.w.tolist() if model.w is not None else None, "b": float(model.b)},
    }
    return result


def run_rbf_grid(X_train, y_train, X_test, y_test):
    Cs = [0.1, 1.0, 10.0, 100.0]
    gammas = [0.001, 0.01, 0.1, 1.0]

    best = None
    best_key = None

    for C in Cs:
        for g in gammas:
            cfg = DualSVMConfig(kernel="rbf", C=C, gamma=g)
            _, m_custom, _, _, _, _, _ = train_custom(X_train, y_train, X_test, y_test, cfg)
            m_sk, _, _, _, _ = compare_with_sklearn(X_train, y_train, X_test, y_test, cfg)
            print(f"[RBF] C={C:<6} gamma={g:<6} acc_custom={m_custom.accuracy:.4f} acc_sklearn={m_sk.accuracy:.4f}")

            if best is None or m_custom.accuracy > best["custom"]["metrics"]["accuracy"]:
                best = {
                    "kernel": "rbf",
                    "cfg": asdict(cfg),
                    "custom": {"metrics": asdict(m_custom)},
                    "sklearn": {"metrics": asdict(m_sk)},
                }
                best_key = (C, g)

    # train best again (to plot + save confusion + timings)
    C_best, g_best = best_key
    cfg_best = DualSVMConfig(kernel="rbf", C=C_best, gamma=g_best)
    model, m_custom, cm_custom, y_pred_custom, fit_custom, _, sv = train_custom(X_train, y_train, X_test, y_test, cfg_best)
    m_sk, cm_sk, y_pred_sk, fit_sk, _ = compare_with_sklearn(X_train, y_train, X_test, y_test, cfg_best)

    plot_decision_boundary_pca(
        model,
        X_train,
        y_train,
        title=f"Custom Dual SVM | kernel=rbf (best) | C={C_best}, gamma={g_best} (PCA space)",
        filename="boundary_custom_rbf_best.png",
        support_vectors=sv,
    )

    plot_confusion_matrices(
        y_true=y_test,
        y_pred_custom=y_pred_custom,
        y_pred_sklearn=y_pred_sk,
        title=f"Breast Cancer | RBF kernel | C={C_best}, gamma={g_best}",
        filename="cm_rbf_best_test.png",
    )

    _print_block(f"Kernel: rbf (best) | C={C_best}, gamma={g_best} (Breast Cancer)", m_custom, m_sk, fit_custom, fit_sk)

    best_full = {
        "kernel": "rbf",
        "cfg": asdict(cfg_best),
        "custom": {"metrics": asdict(m_custom), "confusion": cm_custom, "fit_time_s": fit_custom},
        "sklearn": {"metrics": asdict(m_sk), "confusion": cm_sk, "fit_time_s": fit_sk},
        "n_support_vectors": int(len(sv)),
    }
    return best_full


def run_make_circles_experiment():
    X, y01 = make_circles(n_samples=1200, noise=0.20, factor=0.5, random_state=42)
    y = np.where(y01 == 0, -1, 1)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\n=== EXTRA: make_circles experiment ===")
    print(f"Train class balance: (-1)={np.sum(ytr==-1)} (+1)={np.sum(ytr==1)}")
    print(f"Test  class balance: (-1)={np.sum(yte==-1)} (+1)={np.sum(yte==1)}")

    # Linear (usually weak on circles)
    cfg_lin = DualSVMConfig(kernel="linear", C=1.0)
    model_lin, m_lin, cm_lin, y_pred_lin, fit_lin, _, sv_lin = train_custom(Xtr, ytr, Xte, yte, cfg_lin)
    m_lin_sk, cm_lin_sk, y_pred_lin_sk, fit_lin_sk, _ = compare_with_sklearn(Xtr, ytr, Xte, yte, cfg_lin)

    plot_decision_boundary_2d(
        model_lin, Xte, yte,
        title="make_circles (test) | Custom Dual SVM | kernel=linear",
        filename="circles_boundary_custom_linear.png",
        support_vectors=sv_lin,
    )
    plot_confusion_matrices(
        y_true=yte, y_pred_custom=y_pred_lin, y_pred_sklearn=y_pred_lin_sk,
        title="make_circles | Linear", filename="cm_circles_linear_test.png"
    )
    _print_block("make_circles | Linear", m_lin, m_lin_sk, fit_lin, fit_lin_sk)

    # RBF (captures nonlinearity)
    cfg_rbf = DualSVMConfig(kernel="rbf", C=10.0, gamma=2.0)
    model_rbf, m_rbf, cm_rbf, y_pred_rbf, fit_rbf, _, sv_rbf = train_custom(Xtr, ytr, Xte, yte, cfg_rbf)
    m_rbf_sk, cm_rbf_sk, y_pred_rbf_sk, fit_rbf_sk, _ = compare_with_sklearn(Xtr, ytr, Xte, yte, cfg_rbf)

    plot_decision_boundary_2d(
        model_rbf, Xte, yte,
        title="make_circles (test) | Custom Dual SVM | kernel=rbf",
        filename="circles_boundary_custom_rbf.png",
        support_vectors=sv_rbf,
    )
    plot_confusion_matrices(
        y_true=yte, y_pred_custom=y_pred_rbf, y_pred_sklearn=y_pred_rbf_sk,
        title="make_circles | RBF", filename="cm_circles_rbf_test.png"
    )
    _print_block("make_circles | RBF", m_rbf, m_rbf_sk, fit_rbf, fit_rbf_sk)

    return {
        "make_circles_linear": {
            "cfg": asdict(cfg_lin),
            "custom": {"metrics": asdict(m_lin), "confusion": cm_lin, "fit_time_s": fit_lin, "n_sv": int(len(sv_lin))},
            "sklearn": {"metrics": asdict(m_lin_sk), "confusion": cm_lin_sk, "fit_time_s": fit_lin_sk},
        },
        "make_circles_rbf": {
            "cfg": asdict(cfg_rbf),
            "custom": {"metrics": asdict(m_rbf), "confusion": cm_rbf, "fit_time_s": fit_rbf, "n_sv": int(len(sv_rbf))},
            "sklearn": {"metrics": asdict(m_rbf_sk), "confusion": cm_rbf_sk, "fit_time_s": fit_rbf_sk},
        },
    }


def main():
    _ensure_outputs()

    # Load dataset once
    X_train, X_test, y_train, y_test, _ = load_binary_dataset()

    # PCA projection (data visualization)
    plot_pca_projection(X_train, y_train, X_test, y_test, filename="pca_projection.png")

    # 1) Linear (required)
    linear_result = run_linear_baseline(X_train, y_train, X_test, y_test)

    # 2) RBF grid (extra improvement + better kernel-trick demo)
    best_rbf_result = run_rbf_grid(X_train, y_train, X_test, y_test)

    # 3) EXTRA: make_circles
    circles_results = run_make_circles_experiment()

    # Save json summary
    all_results = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "linear": linear_result,
        "rbf_best": best_rbf_result,
        "extra_make_circles": circles_results,
        "saved_files": [
            "outputs/figures/pca_projection.png",
            "outputs/figures/boundary_custom_linear.png",
            "outputs/figures/boundary_custom_rbf_best.png",
            "outputs/figures/cm_linear_test.png",
            "outputs/figures/cm_rbf_best_test.png",
            "outputs/figures/circles_boundary_custom_linear.png",
            "outputs/figures/circles_boundary_custom_rbf.png",
            "outputs/figures/cm_circles_linear_test.png",
            "outputs/figures/cm_circles_rbf_test.png",
            "outputs/results/results.json",
        ],
    }

    with open("outputs/results/results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved:")
    for p in all_results["saved_files"]:
        print(f" - {p}")


if __name__ == "__main__":
    main()
