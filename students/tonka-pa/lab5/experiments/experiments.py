from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.logreg import MyLogisticRegression, SMGLMLogitClassifier
from src.utils.preprocess import (
    read_alzheimer_dataset,
    classify_cat_cols,
    build_preprocessor,
    score_model_cv,
    compare_models,
)

from . import helpers as exp_helpers

#=========================================================================#

__all__ = [
    "run_tests"
]

RANDOM_SEED = 18012026

#=========================================================================#

def run_tests(
    input_dir: str | Path,
    results_dir: str | Path,
    solvers: list[str] | str | None = None,
):
    df_alzh = read_alzheimer_dataset(input_dir)

    X, y = df_alzh.drop(columns=["Diagnosis"]), df_alzh["Diagnosis"]
    print(f"Dataset shape: {df_alzh.shape}")
    print(f"Number of classes: {df_alzh['Diagnosis'].nunique()}\n")
    print("Classes distribution: ")
    print(y.value_counts(normalize=True), "\n")

    int_cols = X.select_dtypes(include=["int"]).columns.tolist()
    float_cols = X.select_dtypes(include=["float"]).columns.tolist()

    print("Int cols: ", len(int_cols))
    print("Float cols: ", len(float_cols))

    bin_cols, cat_cols = classify_cat_cols(X, int_cols)

    preprocessor = build_preprocessor(int_cols, float_cols, bin_cols, cat_cols)
    X_prep = preprocessor.fit_transform(X)

    # ------------------------------------

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    solver_names = normalize_solver_names(solvers)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    total_variants = sum(len(build_solver_variants(name)) for name in solver_names)
    variant_idx = 0

    for solver_name in solver_names:
        variants = build_solver_variants(solver_name)
        for params in variants:
            variant_idx += 1
            variant_tag = exp_helpers.build_variant_tag(solver_name, params)
            solver_results_dir = results_dir / variant_tag
            solver_results_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 78)
            print(f"Experiment {variant_idx}/{total_variants} | {variant_tag}")
            print("=" * 78)

            print({"solver": solver_name, **params})
            my_params = dict(params)
            my_params["solver"] = solver_name
            my_logreg = MyLogisticRegression(**my_params)
            sm_logreg = SMGLMLogitClassifier(solver=solver_name)

            sk_params = {}
            if "C" in params:
                sk_params["C"] = params["C"]
            sk_logreg = LogisticRegression(**sk_params)

            my_pipe = Pipeline([("preprocess", preprocessor), ("model", my_logreg)])
            sk_pipe = Pipeline([("preprocess", preprocessor), ("model", sk_logreg)])
            sm_pipe = Pipeline([("preprocess", preprocessor), ("model", sm_logreg)])

            print("[Stage] Cross-validation scoring")
            my_metrics = score_model_cv(my_pipe, X, y, cv=cv, include_timing=True)
            sk_metrics = score_model_cv(sk_pipe, X, y, cv=cv, include_timing=True)
            sm_metrics = score_model_cv(sm_pipe, X, y, cv=cv, include_timing=True)

            print("[Stage] Fit full models on preprocessed data")
            my_logreg.fit(X_prep, y)
            sk_logreg.fit(X_prep, y)
            sm_logreg.fit(X_prep, y)

            print("[Stage] Decision boundary plots")
            exp_helpers.plot_decision_boundary(
                my_logreg,
                X_prep,
                y,
                solver_name,
                solver_results_dir,
                use_std=False,
                plot_confidence=True,
                plot_all_points=True,
                title=f"MyLogisticRegression | {variant_tag}",
                variant_tag=variant_tag,
            )
            exp_helpers.plot_decision_boundary(
                sk_logreg,
                X_prep,
                y,
                solver_name,
                solver_results_dir,
                use_std=False,
                plot_confidence=True,
                plot_all_points=True,
                title=f"sklearn LogisticRegression | {variant_tag}",
                variant_tag=variant_tag,
            )
            exp_helpers.plot_decision_boundary(
                sm_logreg,
                X_prep,
                y,
                solver_name,
                solver_results_dir,
                use_std=False,
                plot_confidence=True,
                plot_all_points=True,
                title=f"SMGLMLogitClassifier | {variant_tag}",
                variant_tag=variant_tag,
            )

            solver_results: dict = {}
            exp_helpers.record_result(solver_results, "MyLogisticRegression", my_metrics)
            exp_helpers.record_result(solver_results, "sklearn LogisticRegression", sk_metrics)
            exp_helpers.record_result(solver_results, "SMGLMLogitClassifier", sm_metrics)

            print("[Stage] Metrics comparison report")
            compare_models(
                solver_results,
                sort_by="f1",
                save_dir=solver_results_dir,
                filename=exp_helpers.make_comparison_filename(variant_tag),
            )

            print(f"[Done] Results saved to: {solver_results_dir}")

        # === END of single solver tests


def normalize_solver_names(
    solvers_input: list[str] | str | None
) -> list[str]:
    allowed = {"newton", "irls"}

    if solvers_input is None:
        return ["newton", "irls"]
    if isinstance(solvers_input, list) and not solvers_input:
        return ["newton", "irls"]

    if isinstance(solvers_input, str):
        solvers_list = [solvers_input]
    else:
        solvers_list = list(solvers_input)

    names: list[str] = []
    for item in solvers_list:
        raw_names = [str(item)]
        for raw in raw_names:
            parts = [p for p in raw.replace(",", " ").split() if p]
            for part in parts:
                key = part.strip().lower()
                if key not in allowed:
                    raise ValueError(f"Unknown solver: {part}. Allowed: {sorted(allowed)}")
                if key not in names:
                    names.append(key)
    return names


def build_solver_variants(solver_name: str) -> list[dict]:
    variants = [{"C": 1.0}]
    if solver_name == "newton":
        variants.append({"C": np.inf})
    elif solver_name == "irls":
        variants.append({"C": 0.01})
    return variants
