import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_workflow import load_and_prepare_data, scale_features, train_test_split_data, add_bias_column
from classifier import LinearClassifier
from metrics import Metrics
from graphics import plot_Q_plot, plot_pca_scatter, plot_decision_boundaries, plot_margins
from pca import compute_pca

def model_workflow(name, X_train, X_test, y_train, y_test, multistart=False, **kwargs):
    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    model = LinearClassifier(**kwargs)

    if multistart:
        model.multistart_fit(X_train, y_train, n_restarts=5)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    Metrics.print_all(name, y_test, y_pred)

    plot_Q_plot(model.Q_plot, model.epochs)

    plot_margins(y_train * (np.dot(X_train, model.w).flatten()))

    return model.w[:-1], model.w[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specified LinearClassifier model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["multistart", "margin", "corr"],
        default="corr",
        help="Which model to run: multistart, margin, corr"
    )

    args = parser.parse_args()

    df = load_and_prepare_data()

    X = df.drop(columns='target')

    y = df['target']

    X = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    w = b = None

    match args.model:
        case "multistart":
            w, b = model_workflow("Multistart Model", X_train, X_test, y_train, y_test, multistart=True)
        case "margin":
            w, b = model_workflow("Margin Model", X_train, X_test, y_train, y_test, sampling_strategy="margin")
        case "corr":
            w, b = model_workflow("Correlation Model", X_train, X_test, y_train, y_test, init_method="correlation")

    pca, principal_components = compute_pca(X, n_components=2)
    
    fig, ax = plot_pca_scatter(principal_components, y)

    plot_decision_boundaries(ax, pca, w, b, label='Разделяющая прямая', color='red')

    ax.legend()

    plt.show()