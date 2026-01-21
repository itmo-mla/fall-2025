import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from .data import load_breast_cancer_df
from .knn import my_KNN
from .loo import select_k_by_loo
from .prototype_selection import stolp_select_prototypes
from .plots import plot_empirical_risk, plot_pairplot, plot_prototypes_pca, plot_confusion_matrix


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("out", exist_ok=True)


def prepare_data():
    df = load_breast_cancer_df(data_dir="data")

    y = LabelEncoder().fit_transform(df["diagnosis"].values)
    X_raw = df.drop(columns=["diagnosis"]).values  
    X_scaled = StandardScaler().fit_transform(X_raw)

    return df, X_scaled, y


def run():
    ensure_dirs()
    df, X, y = prepare_data()

    plot_pairplot(
        df,
        features=["radius_mean", "texture_mean", "perimeter_mean", "area_mean"],
        target="diagnosis",
        out_dir="out",
        filename="pairplot.png"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1) LOO choose k
    k_max = 30
    k_values = np.arange(1, k_max)  # 1..29
    best_k, risks = select_k_by_loo(X_train, y_train, k_values, mode="parzen_variable")
    print(f"[LOO] best k = {best_k}, min risk = {float(np.min(risks)):.4f}")

    plot_empirical_risk(
        k_values, risks,
        title="LOO empirical risk for different k",
        out_dir="out",
        filename="loo_risk.png"
    )

    # 2) Custom KNN on full train
    myknn = my_KNN(neighbours=best_k, mode="parzen_variable")
    myknn.fit(X_train, y_train)
    y_pred_custom = myknn.predict(X_test)

    print("\n[Custom KNN - full train]")
    print("Accuracy:", accuracy_score(y_test, y_pred_custom))
    print(classification_report(y_test, y_pred_custom))

    cm_custom = confusion_matrix(y_test, y_pred_custom)
    plot_confusion_matrix(
        cm_custom, class_names=["B(0)", "M(1)"],
        title="Confusion Matrix - Custom KNN (full train)",
        out_dir="out",
        filename="cm_custom_full.png"
    )

    # 3) Reference sklearn KNN
    ref = KNeighborsClassifier(n_neighbors=best_k, weights="uniform")
    ref.fit(X_train, y_train)
    y_pred_sklearn = ref.predict(X_test)

    print("\n[Sklearn KNN - reference]")
    print("Accuracy:", accuracy_score(y_test, y_pred_sklearn))
    print(classification_report(y_test, y_pred_sklearn))

    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    plot_confusion_matrix(
        cm_sklearn, class_names=["B(0)", "M(1)"],
        title="Confusion Matrix - Sklearn KNN",
        out_dir="out",
        filename="cm_sklearn.png"
    )

    # 4) Prototype selection (STOLP-like)
    proto_idx = stolp_select_prototypes(
        X_train, y_train,
        max_prototypes=None,
        remove_noise=True,
        noise_threshold=0.0
    )

    Xp = X_train[proto_idx]
    yp = y_train[proto_idx]
    print(f"\n[Prototype selection] prototypes: {len(proto_idx)} / {len(X_train)}")

    plot_prototypes_pca(
        X_train, y_train, proto_idx,
        title="STOLP-like prototype selection (PCA on train)",
        out_dir="out",
        filename="prototypes_pca.png"
    )

    # 5) Custom KNN on prototypes only
    myknn_p = my_KNN(neighbours=best_k, mode="parzen_variable")
    myknn_p.fit(Xp, yp)
    y_pred_proto = myknn_p.predict(X_test)

    print("\n[Custom KNN - prototypes only]")
    print("Accuracy:", accuracy_score(y_test, y_pred_proto))
    print(classification_report(y_test, y_pred_proto))

    cm_proto = confusion_matrix(y_test, y_pred_proto)
    plot_confusion_matrix(
        cm_proto, class_names=["B(0)", "M(1)"],
        title="Confusion Matrix - Custom KNN (prototypes)",
        out_dir="out",
        filename="cm_custom_prototypes.png"
    )

    # 6) Summary table (compact comparison)
    results = pd.DataFrame([
        {
            "Model": "Custom KNN (full)",
            "k": best_k,
            "Train size": len(X_train),
            "Accuracy": accuracy_score(y_test, y_pred_custom),
        },
        {
            "Model": "Sklearn KNN (ref)",
            "k": best_k,
            "Train size": len(X_train),
            "Accuracy": accuracy_score(y_test, y_pred_sklearn),
        },
        {
            "Model": "Custom KNN (prototypes)",
            "k": best_k,
            "Train size": len(Xp),
            "Accuracy": accuracy_score(y_test, y_pred_proto),
        },
    ])

    # nicer formatting
    results["Accuracy"] = results["Accuracy"].map(lambda v: f"{v:.4f}")

    print("\n=== Compact comparison table ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    run()
