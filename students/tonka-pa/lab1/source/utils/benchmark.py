from itertools import cycle
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             roc_curve, auc, confusion_matrix, RocCurveDisplay)

from .visualization import viz_margins
from logistic_regression import LogRegNumpy


# ----------------- confusion matrix visualisation -----------------
def _ensure_class_names(y_true, y_pred, class_names):
    if class_names is None:
        classes = sorted(list(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())))
        class_names = [str(c) for c in classes]
        labels = classes
    else:
        labels = class_names if not np.issubdtype(np.asarray(y_true).dtype, np.integer) else list(range(len(class_names)))
    return class_names, labels

def plot_multiclass_confusion_matrix(
    y_true, y_pred, class_names=None, normalize=True, cmap="Blues", figsize=(10,8), annot=True, title=None,
    display_plot=False, save_path: str = ''
):
    if save_path:
        save_path = Path(save_path + "multiclass_confusion_matrix.png")

    class_names, labels = _ensure_class_names(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        fmt, cbar_label = ".2f", "Proportion"
    else:
        fmt, cbar_label = "d", "Count"

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=.5, linecolor="white", cbar_kws={"label": cbar_label})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or ("Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix"))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    if display_plot:
        plt.show()
    plt.close()

def _per_class_tp_fp_fn_tn_table(y_true, y_pred, class_names=None):
    class_names, labels = _ensure_class_names(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)
    return pd.DataFrame({"TP": tp, "FP": fp, "FN": fn, "TN": tn}, index=class_names)

def plot_tp_fp_fn_tn_table(y_true, y_pred, class_names=None, cmap="YlGnBu", figsize=(6,1.2), title=None,
                           display_plot=False, save_path: str = ''):
    if save_path:
        save_path = Path(save_path + "tp-fp-fn-tn.png")

    df = _per_class_tp_fp_fn_tn_table(y_true, y_pred, class_names)
    plt.figure(figsize=(figsize[0], figsize[1] * len(df)))
    ax = sns.heatmap(df, annot=True, fmt="d", cmap=cmap, cbar=False,
                     linewidths=.5, linecolor="white")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Class")
    ax.set_title(title or "Per-class TP / FP / FN / TN")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    if display_plot:
        plt.show()
    plt.close()

# ----------------- model adapters -----------------
def _try_fit(model, Xtr, ytr, fit_args, Xval=None, yval=None):
    try:
        return model.fit(Xtr, ytr, **(fit_args or {}))
    except TypeError:
        val_set = (Xval, yval) if (Xval is not None and yval is not None) else (None, None)
        return model.fit((Xtr, ytr), val_set, **(fit_args or {}))

def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    df = model.decision_function(X)
    if df.ndim == 1:
        df = np.c_[-df, df]
    e = np.exp(df - df.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ----------------- New pretty table utilities (ASCII) -----------------

def _print_cv_tables(cv_df: pd.DataFrame, *, title="Cross-Validation", 
                     ascii_borders=True, save_path : str = ''):
    if save_path:
        save_path = Path(save_path + "cv_table.md")

    cv_df = cv_df.copy()
    cv_df.index.name = cv_df.index.name or "fold"

    # numeric-only mean row (avoids dtype warnings)
    mean_row = cv_df.mean(numeric_only=True)
    combined = cv_df.copy()
    combined.loc["mean"] = mean_row

    # draw a rule after the last fold row
    split_after = [len(cv_df) - 1]

    _print_box_table(
        combined,
        title=title,
        index_name=cv_df.index.name,
        ascii_borders=ascii_borders,
        split_after_rows=split_after,
    )

    if save_path:
        markdown_table = combined.to_markdown(index=False, floatfmt=".2f")
        with open(save_path, "w", encoding='utf-8') as file:
            file.write(markdown_table)
            file.write('\n')


def _print_box_table(
    df: pd.DataFrame,
    title=None,
    digits=4,
    split_after_rows=None,
    index_name=None,
    ascii_borders=False,
):
    """
    Pretty box table with:
      - numeric headers & values right-aligned
      - index header & values left-aligned
      - robust width calculation
      - optional ASCII borders if Unicode box-drawing looks misaligned
    """
    if title:
        print("\n" + title)
        print("-" * len(title))

    # choose border chars
    if ascii_borders:
        TL, TM, TR = "+", "+", "+"
        ML, MM, MR = "+", "+", "+"
        BL, BM, BR = "+", "+", "+"
        V = "|"
        H = "-"
    else:
        TL, TM, TR = "┌", "┬", "┐"
        ML, MM, MR = "├", "┼", "┤"
        BL, BM, BR = "└", "┴", "┘"
        V = "│"
        H = "─"

    def _is_num(x):
        return isinstance(x, (int, np.integer, float, np.floating)) or (isinstance(x, str) and x.replace(".","",1).isdigit())

    def _fmt_cell(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.{digits}f}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x)

    # copy as object so we fully control string formatting
    df = df.copy()
    df = df.astype(object)

    # per-column "numericness": treat a column numeric if ALL non-blank entries are numeric
    is_num_col = {}
    for c in df.columns:
        col = df[c].map(lambda v: (v is None) or (isinstance(v, float) and np.isnan(v)) or _is_num(v))
        is_num_col[c] = bool(col.all())

    # stringify data for width calc
    data_str = df.map(_fmt_cell)

    # compute widths (header vs content)
    col_widths = {}
    for c in df.columns:
        header = str(c)
        body_w = 0 if data_str.empty else data_str[c].map(len).max()
        col_widths[c] = max(len(header), body_w)

    idx_name = index_name if index_name is not None else (df.index.name or "")
    idx_vals = df.index.astype(str)
    idx_width = max(len(str(idx_name)), (0 if df.index.empty else idx_vals.map(len).max()))

    def _hseg(left, mid, right):
        # each cell prints as " {content:<w} " or " {content:>w} " so add 2
        return left + H * (idx_width + 2) + "".join(
            mid + H * (col_widths[c] + 2) for c in df.columns
        ) + right

    top    = _hseg(TL, TM, TR)
    mid    = _hseg(ML, MM, MR)
    bottom = _hseg(BL, BM, BR)

    # header: index left; numeric headers right; text headers left
    hdr_idx = f"{idx_name:<{idx_width}}"
    hdr_cells = []
    for c in df.columns:
        h = str(c)
        if is_num_col[c]:
            hdr_cells.append(f"{h:>{col_widths[c]}}")
        else:
            hdr_cells.append(f"{h:<{col_widths[c]}}")
    hdr_line = f"{V} {hdr_idx} {V}" + f"{V}".join(f" {h} " for h in hdr_cells) + f"{V}"

    print(top)
    print(hdr_line)
    print(mid)

    split_after = set(split_after_rows or [])
    for i, (idx, row) in enumerate(df.iterrows()):
        idx_cell = f"{str(idx):<{idx_width}}"
        body_cells = []
        for c in df.columns:
            s = _fmt_cell(row[c])
            if is_num_col[c] and s != "":
                body_cells.append(f"{s:>{col_widths[c]}}")
            else:
                body_cells.append(f"{s:<{col_widths[c]}}")
        line = f"{V} {idx_cell} {V}" + f"{V}".join(f" {cell} " for cell in body_cells) + f"{V}"
        print(line)
        if i in split_after and i != len(df) - 1:
            print(mid)

    print(bottom)
    print()


def _print_pretty_classification_report(
    y_true, y_pred, class_names=None, digits=4,
    title="Classification Report (test set)", ascii_borders=True, save_path: str = ""
):
    if save_path:
        save_path = Path(save_path + "classification_report.md")

    rep = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    order_cols = ["precision", "recall", "f1-score", "support"]

    # class rows
    class_labels = class_names if class_names is not None else [
        k for k in rep.keys() if k not in ("accuracy", "macro avg", "weighted avg")
    ]
    df_classes = pd.DataFrame(
        [[rep[l].get(c, "") for c in order_cols] for l in class_labels],
        index=class_labels, columns=order_cols
    )

    # --- FIXED: use empty strings instead of None/NaN in the accuracy row ---
    total_support = int(sum(rep[l]["support"] for l in class_labels))
    acc_row = pd.DataFrame(
        [{"precision": "", "recall": "", "f1-score": rep["accuracy"], "support": total_support}],
        index=["accuracy"]
    )

    df_macro    = pd.DataFrame([[rep["macro avg"][c]    for c in order_cols]], index=["macro avg"],    columns=order_cols)
    df_weighted = pd.DataFrame([[rep["weighted avg"][c] for c in order_cols]], index=["weighted avg"], columns=order_cols)

    df = pd.concat([df_classes, acc_row, df_macro, df_weighted], axis=0)


    # draw a horizontal rule after per-class rows
    split_after = [len(df_classes) - 1]
    _print_box_table(df, title=title, digits=digits, split_after_rows=split_after, ascii_borders=ascii_borders)

    if save_path:
        markdown_table = df.to_markdown(index=False, floatfmt=".2f")
        with open(save_path, "w", encoding='utf-8') as file:
            file.write(markdown_table)
            file.write('\n')


def plot_roc_ovr_sklearn_style(y_train, y_test, y_proba, class_names=None,
                               fig_kw=dict(figsize=(6, 6)),
                               title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
                               display_plot: bool = False, save_path: str = ''):
    """
    Plot OvR ROC curves with the same look & feel as the scikit-learn example:
    - per-class curves via RocCurveDisplay (aqua/darkorange/cornflowerblue cycling)
    - micro-average: deeppink, dotted, lw=4
    - macro-average: navy, dotted, lw=4
    - chance level dashed black
    """
    if save_path:
        save_path = Path(save_path + "roc.png")

    # Binarize
    lb = LabelBinarizer().fit(y_train)
    y_onehot_test = lb.transform(y_test)
    if y_onehot_test.shape[1] == 1:
        # binary edge case -> make it 2 columns
        y_onehot_test = np.c_[1 - y_onehot_test, y_onehot_test]

    n_classes = y_onehot_test.shape[1]
    if class_names is None:
        class_names = [f"class {i}" for i in range(n_classes)]

    # Per-class ROC (to compute macro later)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # ---- Plot (sklearn style) ----
    fig, ax = plt.subplots(**fig_kw)

    # micro
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink", linestyle=":", linewidth=4,
    )

    # macro
    ax.plot(
        fpr["macro"], tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy", linestyle=":", linewidth=4,
    )

    # per-class curves via RocCurveDisplay (cycle colors as in the example)
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_proba[:, class_id],
            name=f"ROC curve for {class_names[class_id]}",
            ax=ax,
            plot_chance_level=(class_id == n_classes - 1),  # dashed black diagonal once
            despine=True,
            **dict(color=color, linewidth=2)
        )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
        # xlim=(0.0, 1.0),
        # ylim=(0.0, 1.05),
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    if display_plot:
        plt.show()
    plt.close()

def plot_loss_per_epoch(model, display_plot: bool = False, save_path: str = ''):
    if save_path:
        save_path = Path(save_path + "loss.png")

    if isinstance(model, LogRegNumpy):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey='row')

        ax[0].spines[['right', 'top']].set_visible(False)
        ax[0].grid(ls='--', alpha=0.6)
        ax[0].plot(model.loss_values[0], label='train')
        ax[0].plot(model.loss_values[1], label='test')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title("Train vs Test loss")
        ax[0].legend()

        ax[1].spines[['right', 'top']].set_visible(False)
        ax[1].grid(ls='--', alpha=0.6)
        ax[1].plot(model.rec_history)
        ax[1].set_xlabel('Step')
        ax[1].set_ylabel('Loss (smoothed)')
        ax[1].set_title("Train (smoothed) loss")
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
        if display_plot:
            plt.show()
        plt.close()
    else:
        print('Model should be of type LogRegNumpy')
    return

# ----------------- One-model benchmark (same interface) -----------------

def benchmark_classifier(
    model_cls,
    X_train, y_train,
    X_test,  y_test,
    *,
    cv_folds=5,
    model_args=None,
    fit_args=None,
    random_state=18092025,
    class_names=None,
    plot_loss=True, display_loss=False,
    plot_roc=True, display_roc=False,
    plot_confusions=True, display_confusions=False,
    plot_margins=True, display_margins=False,
    ascii_borders=True,
    eps=1.0,
    output_path: str = "",
    test_name_suffix: str = ''
):
    print("=" * 40 + ' ' + model_cls.__name__ + "_" + test_name_suffix + ' ' + "=" * 40)

    model_args = model_args or {}
    fit_args   = fit_args   or {}

    if output_path:
        base_path = Path(output_path)
        images_dir = base_path / "images"
        tables_dir = base_path / "tables"
        images_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{model_cls.__name__}_{test_name_suffix}_"
        image_save_path = (images_dir / prefix).as_posix()
        table_save_path = (tables_dir / prefix).as_posix()
    else:
        image_save_path = ""
        table_save_path = ""

    # ===== 1) Stratified CV on train =====
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    per_fold = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]

        model = model_cls(**model_args)

        # timing
        t0 = perf_counter()
        _try_fit(model, Xtr, ytr, fit_args, Xval=Xva, yval=yva)
        fit_time = perf_counter() - t0

        yhat = model.predict(Xva)

        acc  = accuracy_score(yva, yhat)
        p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(yva, yhat, average="micro",    zero_division=0)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(yva, yhat, average="macro",    zero_division=0)
        p_w, r_w, f_w, _             = precision_recall_fscore_support(yva, yhat, average="weighted", zero_division=0)

        per_fold.append(dict(
            fold=fold,
            accuracy=acc,
            precision_micro=p_micro, recall_micro=r_micro, f1_micro=f_micro,
            precision_macro=p_macro, recall_macro=r_macro, f1_macro=f_macro,
            precision_weighted=p_w, recall_weighted=r_w, f1_weighted=f_w,
            fit_time_sec=fit_time,
        ))

    cv_df = pd.DataFrame(per_fold).set_index("fold")
    _print_cv_tables(cv_df, title="Cross-Validation (per-fold + mean)",
                     ascii_borders=ascii_borders, save_path=table_save_path)

    # ===== 2) Fit on full train, evaluate on test =====
    final_model = model_cls(**model_args)

    t0 = perf_counter()
    _try_fit(final_model, X_train, y_train, fit_args)
    final_fit_time = perf_counter() - t0

    y_pred  = final_model.predict(X_test)
    y_proba = _proba(final_model, X_test)

    _print_pretty_classification_report(
        y_test, y_pred, class_names=class_names, digits=4,
        title=f"Classification Report (test set)  [final fit {final_fit_time:.2f}s]",
        ascii_borders=ascii_borders, save_path=table_save_path
    )

    # ===== 3) Plots (Loss + ROC + Confusions + Margins) =====
    if plot_loss:
        plot_loss_per_epoch(final_model, display_loss, save_path=image_save_path)
    
    if plot_roc:
        plot_roc_ovr_sklearn_style(y_train, y_test, y_proba, class_names=class_names,
                                   display_plot=display_roc, save_path=image_save_path)

    if plot_confusions:
        plot_multiclass_confusion_matrix(
            y_test, y_pred, class_names=class_names, normalize=True,
            title="Confusion Matrix (test, row-normalized)",
            display_plot=display_confusions, save_path=image_save_path
        )
        plot_tp_fp_fn_tn_table(
            y_test, y_pred, class_names=class_names,
            title="Per-class TP/FP/FN/TN (test)",
            display_plot=display_confusions, save_path=image_save_path
        )

    # Optional margin curves (only if model implements `calc_margins`)
    if plot_margins and hasattr(final_model, "calc_margins") and callable(getattr(final_model, "calc_margins")):
        try:
            # Train margins (on full train the model was fit on)
            train_margins = final_model.calc_margins(X_train, y_train)
            viz_margins(train_margins, eps=eps, display_plot=display_margins,
                        save_path=image_save_path)
            # Test margins
            test_margins  = final_model.calc_margins(X_test,  y_test)
            viz_margins(test_margins, eps=eps, display_plot=display_margins,
                        save_path=image_save_path)
        except Exception as e:
            # Be silent but informative
            print(f"[info] Skipped margin plots due to error: {e}")

    # ===== 4) Return results (kept compatible) =====
    results = {
        "cv_per_fold": cv_df,                              # per-fold metrics (+ fit_time_sec)
        "cv_mean": cv_df.mean(numeric_only=True).to_frame().T,  # one-row DataFrame
        "y_test": y_test,
        "y_pred": y_pred,
        "final_fit_time_sec": final_fit_time,              # extra convenience
    }
    print("="*100 + '\n')
    return final_model, results


if __name__ == "__main__":
    #TODO: add behaviour of calling this script directly
    pass
