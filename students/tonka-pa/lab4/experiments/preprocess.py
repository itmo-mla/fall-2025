from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

#==================================================================

__all__ = [
    "preprocess_dataset",
    "plot_cor_mat",
]

RANDOM_SEED = 23012026

#==================================================================

def preprocess_dataset(
    df: pd.DataFrame,
    *,
    sample_size: int = 10000,
    results_dir: Path | str | None = None,
    correlation_plot_name: str = "correlation_matrix_pearson.png",
):
    int_cols = df.select_dtypes(include=['int']).columns
    float_cols = df.select_dtypes(include=['float']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    float_cols = float_cols.drop(labels=['user_engagement_score'], errors='ignore')
    int_cols = int_cols.drop(labels=['user_engagement_score'], errors='ignore')

    ord_cols = {
        'diet_quality': ['Very poor', 'Poor', 'Average', 'Good', 'Excellent'],
        'income_level': ['Low', 'Lower-middle', 'Middle', 'Upper-middle', 'High'],
        'education_level': ['Other', 'High school', 'Some college', 'Bachelor’s', 'Master’s', 'PhD'],
        'smoking': ['No', 'Former', 'Yes'],
        'alcohol_frequency': ['Never', 'Rarely', 'Weekly', 'Several times a week', 'Daily'],
        'privacy_setting_level': ['Private', 'Friends only', 'Public'],
    }
    ord_cols = {k: v for k, v in ord_cols.items() if k in df.columns}
    cat_cols = cat_cols.drop(ord_cols.keys(), errors='ignore')

    corr_cols = list(int_cols) + list(float_cols)
    if 'user_engagement_score' in df.columns:
        corr_cols.append('user_engagement_score')
    pearson_corr = df[corr_cols].corr(method='pearson')
    save_path = None
    if results_dir is not None:
        save_path = Path(results_dir) / correlation_plot_name
    plot_cor_mat(pearson_corr, title='Correlation matrix (Pearson)', save_path=save_path)

    if sample_size and len(df) > sample_size:
        df_samp = df.sample(n=sample_size, random_state=RANDOM_SEED)
    else:
        df_samp = df.copy()

    X, y = df_samp.drop(columns=['user_engagement_score']), df_samp['user_engagement_score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=RANDOM_SEED
    )
    print("Train/test sizes: ", X_train.shape, X_test.shape)

    X       = X.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y       = y.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    numeric_features = list(int_cols) + list(float_cols)
    ordinal_features = list(ord_cols.keys())
    categorical_features = [c for c in cat_cols if c not in ordinal_features]

    numeric_transformer = Pipeline(
        steps=[("scale", StandardScaler())]
    )
    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ord_cols[c] for c in ordinal_features],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist'))]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if ordinal_features:
        transformers.append(("ord", ordinal_transformer, ordinal_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return X, y, X_train, X_test, y_train, y_test, preprocessor


#==================================================================

#========== Correlation plot ==========#

def plot_cor_mat(
    correlation_matrix, 
    mask=True, 
    colormap='coolwarm', 
    title='', 
    params={}, 
    show=False, 
    save_path: Path | str = None
):
    plt.figure(figsize=(12, 10))

    plot_params = dict(
        linewidth=0.5, 
        linecolor='white'
    )
    plot_params.update(**params)

    if mask:
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, mask=mask, annot=True, cmap=colormap,
            fmt='.2f', cbar_kws={"shrink": 0.75}, annot_kws={"fontsize": "x-small"}, **plot_params
            )
    else:
        sns.heatmap(
            correlation_matrix, annot=True, cmap=colormap, fmt='.2f',
            cbar_kws={"shrink": 0.75}, annot_kws={"fontsize": "x-small"}, **plot_params
            )
        
    plt.title(title, fontsize=16)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    if show:
        plt.show()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
