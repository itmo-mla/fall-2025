from logres import LogisticRegression as CustomLogisticRegression
from metrics import Metrics
from etalon import sklearn_model


def custom_model(X_train, X_test, y_train, y_test):

    model = CustomLogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    Metrics.print_all("Custom Model", y_test, y_pred)

    return model.coef_, model.intercept_


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

    w, b = custom_model(X_train, X_test, y_train, y_test)
    w2, b2 = sklearn_model(X_train, X_test, y_train, y_test)

    pca, principal_components = compute_pca(X, n_components=2)
    
    fig, ax = plot_pca_scatter(principal_components, y)

    plot_decision_boundaries(ax, pca, w, b, label='Разделяющая прямая custom', color='red')
    plot_decision_boundaries(ax, pca, w2, b2, label='Разделяющая прямая sklearn', color='green')

    print("coef_ custom: ", w)
    print("intercept_ custom: ", b)

    print("coef_ sklearn: ", w2)
    print("intercept_ sklearn: ", b2)

    ax.legend()

    plt.show()