from sklearn.linear_model import LogisticRegression
from metrics import Metrics


def sklearn_model(X_train, X_test, y_train, y_test):

    model = LogisticRegression(penalty=None, solver='newton-cholesky')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    Metrics.print_all("Sklearn Model", y_test, y_pred)

    return model.coef_[0], model.intercept_[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
    from pca import compute_pca
    from graphics import plot_decision_boundaries, plot_pca_scatter

    df = load_and_prepare_data()

    X = df.drop(columns='target')

    y = df['target']

    X = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    w, b = sklearn_model(X_train, X_test, y_train, y_test)

    pca, principal_components = compute_pca(X, n_components=2)
    
    fig, ax = plot_pca_scatter(principal_components, y)

    plot_decision_boundaries(ax, pca, w, b, label='Разделяющая прямая', color='red')

    print("coef_: ", w)
    print("intercept_: ", b)

    ax.legend()

    plt.show()