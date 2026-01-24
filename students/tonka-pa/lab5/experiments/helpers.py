from pathlib import Path
from typing import Any, Mapping

from sklearn.linear_model import LogisticRegression

from src.logreg import MyLogisticRegression, SMGLMLogitClassifier
from src.utils.viz import plot_logreg_solution_lda_pca2, plot_clf_solution_umap2_fit

#=========================================================================#

__all__ = [
    "make_decision_boundary_filename",
    "make_comparison_filename",
    "build_param_tag",
    "build_variant_tag",
    "record_result",
    "plot_decision_boundary",
]

#=========================================================================#

def _slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "-")
        .replace("+", "-")
        .replace("_", "-")
    )

def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


DEFAULT_LOGREG_PARAMS = {
    "C": 1.0,
}

PARAM_TAGS = {
    "C": "c",
}


def build_param_tag(params: Mapping[str, Any]) -> str:
    tags: list[str] = []
    for key, abbr in PARAM_TAGS.items():
        if key not in params:
            continue
        value = params[key]
        default = DEFAULT_LOGREG_PARAMS.get(key)
        if value != default:
            tags.append(f"{abbr}-{_slugify(_format_param_value(value))}")
    return "_".join(tags)


def build_variant_tag(solver_name: str, params: Mapping[str, Any]) -> str:
    solver_tag = f"solver-{_slugify(solver_name)}"
    param_tag = build_param_tag(params)
    if param_tag:
        return f"{solver_tag}_{param_tag}"
    return solver_tag

def make_decision_boundary_filename(
    variant_tag: str,
    run_tag: str | int,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool
) -> str:
    variant_slug = _slugify(str(variant_tag))
    run_tag = _slugify(str(run_tag)) if run_tag is not None else "plot"
    conf_tag = "confidence" if plot_confidence else "labels"
    std_tag = "std" if use_std else "raw"
    points_tag = "all" if plot_all_points else "subset"
    return f"{run_tag}_{variant_slug}_{conf_tag}_{std_tag}_{points_tag}.png"


def make_comparison_filename(variant_tag: str) -> str:
    return f"model_comparison_{_slugify(variant_tag)}.md"


def _model_tag(
    clf: MyLogisticRegression | SMGLMLogitClassifier | LogisticRegression,
) -> str:
    if isinstance(clf, MyLogisticRegression):
        return "mylogreg"
    if isinstance(clf, SMGLMLogitClassifier):
        return "smglm-logit"
    return "sklearn-logreg"

#=========================================================================#
# Experiments helpers
#=========================================================================#

def record_result(results: dict, name: str, metrics: dict) -> None:
    results[name] = dict(metrics)


def plot_decision_boundary(
    clf: MyLogisticRegression | SMGLMLogitClassifier | LogisticRegression,
    X,
    y,
    solver_name: str,
    solver_results_dir: Path,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool,
    title: str = "",
    variant_tag: str | None = None,
) -> None:
    model_tag = _model_tag(clf)
    if variant_tag is None:
        variant_tag = build_variant_tag(solver_name, clf.get_params())

    lda_filename = make_decision_boundary_filename(
        variant_tag,
        f"{model_tag}_lda-pca2",
        use_std,
        plot_confidence,
        plot_all_points,
    )
    plot_logreg_solution_lda_pca2(
        estimator=clf,
        X=X,
        y=y,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
        title=title,
        save_dir=solver_results_dir,
        filename=lda_filename,
    )

    umap_filename = make_decision_boundary_filename(
        variant_tag,
        f"{model_tag}_umap2-fit",
        use_std,
        plot_confidence,
        plot_all_points,
    )
    plot_clf_solution_umap2_fit(
        estimator=clf,
        X=X,
        y=y,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
        title=title,
        save_dir=solver_results_dir,
        filename=umap_filename,
    )

#=========================================================================#
# End experiments helpers
#=========================================================================#
