from pathlib import Path
from typing import Any, Mapping

from sklearn.svm import SVC

from src.svc import MySVC
from src.utils.viz import plot_svc_solution_lda_pca2, plot_svc_solution_umap2_fit

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


DEFAULT_SVC_PARAMS = {
    "C": 1.0,
    "degree": 3,
    "gamma": "scale",
    "coef0": 0.0,
}

PARAM_TAGS = {
    "C": "c",
    "degree": "deg",
    "gamma": "gamma",
    "coef0": "coef0",
}


def build_param_tag(params: Mapping[str, Any]) -> str:
    tags: list[str] = []
    for key, abbr in PARAM_TAGS.items():
        if key not in params:
            continue
        value = params[key]
        default = DEFAULT_SVC_PARAMS.get(key)
        if value != default:
            tags.append(f"{abbr}-{_slugify(_format_param_value(value))}")
    return "_".join(tags)


def build_variant_tag(kernel_name: str, params: Mapping[str, Any]) -> str:
    kernel_tag = f"kernel-{_slugify(kernel_name)}"
    param_tag = build_param_tag(params)
    if param_tag:
        return f"{kernel_tag}_{param_tag}"
    return kernel_tag

def make_decision_boundary_filename(
    kernel_name: str,
    k: str | int,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool
) -> str:
    kernel_tag = _slugify(str(kernel_name))
    run_tag = _slugify(str(k)) if k is not None else "plot"
    conf_tag = "confidence" if plot_confidence else "labels"
    std_tag = "std" if use_std else "raw"
    points_tag = "all" if plot_all_points else "sv"
    return f"{run_tag}_{kernel_tag}_{conf_tag}_{std_tag}_{points_tag}.png"


def make_comparison_filename(variant_tag: str) -> str:
    return f"model_comparison_{_slugify(variant_tag)}.md"


def _model_tag(clf: SVC | MySVC) -> str:
    if isinstance(clf, MySVC):
        return "mysvc"
    return "sklearn-svc"

#=========================================================================#
# Experiments helpers
#=========================================================================#

def record_result(results: dict, name: str, metrics: dict, n_prototypes: int) -> None:
    metrics["number of support vectors"] = n_prototypes
    results[name] = metrics


def plot_decision_boundary(
    clf: SVC | MySVC,
    X,
    y,
    kernel_name: str,
    kernel_results_dir: Path,
    use_std: bool,
    plot_confidence: bool,
    plot_all_points: bool,
    title: str = "",
    variant_tag: str | None = None,
) -> None:
    model_tag = _model_tag(clf)
    if variant_tag is None:
        variant_tag = build_variant_tag(kernel_name, clf.get_params())

    lda_filename = make_decision_boundary_filename(
        variant_tag,
        f"{model_tag}_lda-pca2",
        use_std,
        plot_confidence,
        plot_all_points,
    )
    plot_svc_solution_lda_pca2(
        estimator=clf,
        X=X,
        y=y,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
        title=title,
        save_dir=kernel_results_dir,
        filename=lda_filename,
    )

    umap_filename = make_decision_boundary_filename(
        variant_tag,
        f"{model_tag}_umap2-fit",
        use_std,
        plot_confidence,
        plot_all_points,
    )
    plot_svc_solution_umap2_fit(
        estimator=clf,
        X=X,
        y=y,
        use_std=use_std,
        plot_confidence=plot_confidence,
        plot_all_points=plot_all_points,
        title=title,
        save_dir=kernel_results_dir,
        filename=umap_filename,
    )

#=========================================================================#
# End experiments helpers
#=========================================================================#
