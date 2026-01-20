from time import perf_counter_ns
from typing import Union, Optional, Any, Literal, Mapping, Iterable
from pathlib import Path
from collections import defaultdict

import zipfile
import requests

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

#================================================================================================================#

__all__ = [
    "check_for_alzheimers_dataset",
    "read_alzheimer_dataset",
    "classify_cat_cols",
    "build_preprocessor",
    "score_model_cv",
    "compare_models"
]

#================================================================================================================#

RANDOM_SEED = 18012026

#========== Download Dataset ==========#

def check_for_alzheimers_dataset(
    *,
    filename: str = "alzheimers_disease_data.csv",
    force_download: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    # Resolve project root from this file location:
    # utils.py -> utils -> src -> project-root
    project_root = Path(__file__).resolve().parents[2]
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = datasets_dir / filename
    if csv_path.exists() and not force_download:
        return csv_path

    url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "rabieelkharoua/alzheimers-disease-dataset"
    )
    zip_path = datasets_dir / "alzheimers-disease-dataset.zip"

    # Download ZIP (streaming)
    try:
        with requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download dataset from Kaggle endpoint: {e}") from e

    # Extract ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(datasets_dir)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "Downloaded file is not a valid ZIP. "
            "Kaggle may have returned an HTML page or an auth/consent response."
        ) from e
    finally:
        # Clean up zip if it exists
        if zip_path.exists():
            zip_path.unlink()

    if not csv_path.exists():
        # If Kaggle changes the CSV name, show what's in datasets_dir to help debug.
        candidates = sorted(
            p.name for p in datasets_dir.glob("*.csv") if p.is_file()
        )
        raise FileNotFoundError(
            f"Expected CSV '{filename}' not found in {datasets_dir} after extraction. "
            f"CSV files present: {candidates or 'none'}"
        )

    return csv_path

#========== Preprocessing ==========#

def read_alzheimer_dataset(
    input_dir: str | Path
) -> pd.DataFrame:
    df = pd.read_csv(input_dir, header=0)
    df = df.drop(columns=["PatientID", "DoctorInCharge"])
    return df


def classify_cat_cols(X: pd.DataFrame, int_cols: list) -> tuple[list, list]:
    bin_cols = []
    cat_cols = []
    for col in int_cols:
        n_unique = X[col].nunique()
        if n_unique == 2:
            bin_cols.append(col)
        elif n_unique > 2 and n_unique < 10:
            cat_cols.append(col)

    for col in (bin_cols + cat_cols):
        int_cols.remove(col)

    print("int cols:         ", len(int_cols))
    print("catecorical cols: ", len(cat_cols))
    print("binary cols:      ", len(bin_cols), "\n")

    return bin_cols, cat_cols


def build_preprocessor(
    int_cols: list,
    float_cols: list,
    bin_cols: list,
    cat_cols: list
) -> ColumnTransformer:
    preprocessing = ColumnTransformer(
        transformers=[
            ("float_cols", StandardScaler(), float_cols + int_cols + [cat_cols[1]]),
            ("cat_cols", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), [cat_cols[0]]),
            # ("ord_cols", OrdinalEncoder(), [cat_cols[1]]),
            ("bin_cols", "passthrough", bin_cols),
        ],
        remainder="passthrough"
    )
    return preprocessing



#========== Scorer ==========#

def score_model_cv(
    estimator,
    X,
    y,
    *,
    cv_splits: int = 5,
    cv: StratifiedKFold | None = None,
    scoring: Mapping[str, str] | None = None,
    n_jobs: int | None = None,
    include_timing: bool = False,
) -> dict:
    if scoring is None:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

    if cv is None:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=n_jobs,
    )

    metrics = {}
    for metric in scoring:
        scores = results.get(f"test_{metric}")
        if scores is not None:
            metrics[metric] = float(np.mean(scores))

    if include_timing:
        fit_time = results.get("fit_time")
        if fit_time is not None:
            metrics["fit time"] = float(np.mean(fit_time))

    return metrics

#================================================================================================================#

#========== Models comparison based on results dict ===========#

def compare_models(
    results: dict,
    sort_by: Literal['f1', 'accuracy', 'precision', 'recall', 'fit time'] = 'f1',
    save_dir: Path | None = None,
    filename: str = "model_comparison.md",
):
    key_map = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "fit time": "Fit Time (s)",
        "number of support vectors": "Number of Support Vectors",
        "number of prototypes": "Number of Support Vectors",
    }

    rows = []
    for model_name, metrics in results.items():
        # metrics might be a defaultdict(list, {...}); coerce to plain dict
        metrics_dict = dict(metrics)
        # normalize keys (lowercase, trimmed)
        normalized = {str(k).strip().lower(): v for k, v in metrics_dict.items()}

        row = {"Model": model_name}
        for in_key, out_col in key_map.items():
            if in_key in normalized:
                row[out_col] = normalized[in_key]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist (even if missing in some models)
    cols_order = [
        "Model",
        "F1",
        "Accuracy",
        "Precision",
        "Recall",
        "Fit Time (s)",
        "Number of Support Vectors",
    ]
    for c in cols_order:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce numeric columns for sorting/styling
    numeric_cols = [c for c in cols_order if c != "Model"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Sort by requested metric (descending)
    df = df[cols_order].sort_values(by=key_map[sort_by], ascending=False, na_position="last").reset_index(drop=True)

    # Save to Markdown if directory is provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        md_path = save_dir / filename
        df.to_markdown(md_path, index=False)

    # Metrics groups
    higher_is_better = ["F1", "Accuracy", "Precision", "Recall"]
    lower_is_better = ["Fit Time (s)", "Number of Support Vectors"]

    styled_df = (
        df.style
        # Best (green) / worst (red)
        .highlight_max(subset=higher_is_better, color="#c6efce")
        .highlight_min(subset=lower_is_better, color="#c6efce")
        .highlight_min(subset=higher_is_better, color="#ffc7ce")
        .highlight_max(subset=lower_is_better, color="#ffc7ce")
        # Gradients
        .background_gradient(subset=higher_is_better, cmap="RdYlGn")
        .background_gradient(subset=lower_is_better, cmap="RdYlGn_r")
        # Formatting
        .format({
            "F1": "{:.3f}",
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "Fit Time (s)": "{:.3f}",
            "Number of Support Vectors": "{}"
        }, na_rep="—")
        .set_caption("Model Performance Comparison (sorted by F1)")
    )

    return styled_df

#================================================================================================================#
