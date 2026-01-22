from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogReg

from .logreg import LogisticRegressionLecture
from .metrics import accuracy, confusion_matrix_pm, roc_curve_from_scores, auc_trapz
from .plots import plot_convergence_Q, plot_confusion_matrix, plot_roc
from .data import DatasetSplit


@dataclass
class Report:
    name: str
    converged: bool
    n_iter: int
    acc_train: float
    acc_test: float
    cm_test: np.ndarray
    proba_test: np.ndarray
    w: np.ndarray


def _fit_sklearn_reference(X_train: np.ndarray, y_train_pm: np.ndarray) -> SkLogReg:
    """
    Reference implementation (allowed).

    To avoid sklearn 1.8+ warnings:
    - do NOT set penalty=None
    - keep default penalty='l2'
    - use a very large C to make regularization negligible
    """
    y01 = (y_train_pm == 1).astype(int)

    model = SkLogReg(
        # leave penalty at default ('l2') -> no FutureWarning / UserWarning
        C=1e12,            # extremely weak regularization ~ "almost none"
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_train, y01)
    return model



def run_experiment(
    ds: DatasetSplit,
    out_dir: Path,
    max_iter: int = 200,
    tol: float = 1e-8,
    q_tol: float = 1e-10,   
    ridge: float = 1e-10,
    step_size: float = 1.0,
) -> dict[str, Report]:
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Xte = ds.X_train, ds.X_test
    ytr, yte = ds.y_train, ds.y_test

    reports: dict[str, Report] = {}

    # --- Lecture-Newton ---
    m_newton = LogisticRegressionLecture(
        method="newton",
        max_iter=max_iter,
        tol=tol,
        q_tol=q_tol,     
        ridge=ridge,
        step_size=step_size,
    )
    fit_newton = m_newton.fit(Xtr, ytr)
    pred_tr = m_newton.predict(Xtr)
    pred_te = m_newton.predict(Xte)
    proba_te = m_newton.predict_proba(Xte)[:, 1]
    reports["Lecture-Newton"] = Report(
        name="Lecture-Newton",
        converged=fit_newton.converged,
        n_iter=fit_newton.n_iter,
        acc_train=accuracy(ytr, pred_tr),
        acc_test=accuracy(yte, pred_te),
        cm_test=confusion_matrix_pm(yte, pred_te),
        proba_test=proba_te,
        w=fit_newton.w,
    )

    # --- Lecture-IRLS ---
    m_irls = LogisticRegressionLecture(
        method="irls",
        max_iter=max_iter,
        tol=tol,
        q_tol=q_tol,      # <-- ADD
        ridge=ridge,
        step_size=step_size,
    )
    fit_irls = m_irls.fit(Xtr, ytr)
    pred_tr2 = m_irls.predict(Xtr)
    pred_te2 = m_irls.predict(Xte)
    proba_te2 = m_irls.predict_proba(Xte)[:, 1]
    reports["Lecture-IRLS"] = Report(
        name="Lecture-IRLS",
        converged=fit_irls.converged,
        n_iter=fit_irls.n_iter,
        acc_train=accuracy(ytr, pred_tr2),
        acc_test=accuracy(yte, pred_te2),
        cm_test=confusion_matrix_pm(yte, pred_te2),
        proba_test=proba_te2,
        w=fit_irls.w,
    )

    # --- Sklearn reference ---
    sk = _fit_sklearn_reference(Xtr, ytr)
    yte01 = (yte == 1).astype(int)
    sk_pred01 = sk.predict(Xte)
    sk_pred_pm = np.where(sk_pred01 == 1, 1, -1)
    sk_proba_pos = sk.predict_proba(Xte)[:, 1]

    # sklearn weights in our lecture layout: [coef..., bias_last]
    w_ref = np.concatenate([sk.coef_.reshape(-1), sk.intercept_.reshape(-1)])

    reports["Sklearn-Reference"] = Report(
        name="Sklearn-Reference",
        converged=True,
        n_iter=int(getattr(sk, "n_iter_", [0])[0]) if hasattr(sk, "n_iter_") else 0,
        acc_train=accuracy(ytr, np.where(sk.predict(Xtr) == 1, 1, -1)),
        acc_test=accuracy(yte, sk_pred_pm),
        cm_test=confusion_matrix_pm(yte, sk_pred_pm),
        proba_test=sk_proba_pos,
        w=w_ref,
    )

    # --- Save equivalence ---
    wN = reports["Lecture-Newton"].w
    wI = reports["Lecture-IRLS"].w
    wS = reports["Sklearn-Reference"].w

    with (out_dir / "equivalence.txt").open("w", encoding="utf-8") as f:
        f.write("Equivalence (lecture formulas, y in {-1,+1})\n\n")
        f.write(f"||w_newton - w_irls||_2 = {float(np.linalg.norm(wN - wI)):.12e}\n")
        f.write(f"max|w_newton - w_irls|  = {float(np.max(np.abs(wN - wI))):.12e}\n\n")
        f.write("Comparison with sklearn reference (may differ slightly due to solver/regularization)\n")
        f.write(f"||w_newton - w_sklearn||_2 = {float(np.linalg.norm(wN - wS)):.12e}\n")
        f.write(f"max|w_newton - w_sklearn|  = {float(np.max(np.abs(wN - wS))):.12e}\n")

    # --- Save histories as CSV ---
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Q": fit_newton.history["Q"], "step_norm": fit_newton.history["step_norm"]}).to_csv(
        hist_dir / "history_newton.csv", index_label="iter"
    )
    pd.DataFrame({"Q": fit_irls.history["Q"], "step_norm": fit_irls.history["step_norm"]}).to_csv(
        hist_dir / "history_irls.csv", index_label="iter"
    )

    # --- Save metrics summary ---
    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for k, r in reports.items():
            f.write(f"{k}\n")
            f.write(f"  converged={r.converged}, n_iter={r.n_iter}\n")
            f.write(f"  acc_train={r.acc_train:.4f}, acc_test={r.acc_test:.4f}\n")
            f.write(f"  cm_test=\n{r.cm_test}\n\n")

    # --- Plots ---
    plot_convergence_Q(
        {
            "Newton (lecture)": fit_newton.history,
            "IRLS (lecture)": fit_irls.history,
        },
        out_path=out_dir / "convergence_Q.png",
    )

    plot_confusion_matrix(
        reports["Lecture-Newton"].cm_test,
        "Confusion matrix (Newton, lecture) on test",
        out_dir / "cm_newton.png",
    )
    plot_confusion_matrix(
        reports["Lecture-IRLS"].cm_test,
        "Confusion matrix (IRLS, lecture) on test",
        out_dir / "cm_irls.png",
    )
    plot_confusion_matrix(
        reports["Sklearn-Reference"].cm_test,
        "Confusion matrix (sklearn reference) on test",
        out_dir / "cm_sklearn.png",
    )

    curves = {}
    for k in ["Lecture-Newton", "Lecture-IRLS", "Sklearn-Reference"]:
        fpr, tpr, _ = roc_curve_from_scores(yte, reports[k].proba_test)
        curves[k] = (fpr, tpr, auc_trapz(fpr, tpr))
    plot_roc(curves, out_dir / "roc.png")

    return reports
