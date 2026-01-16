from sklearn.svm import SVC
from metrics import Metrics
import matplotlib.pyplot as plt
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
from pca import compute_pca
from graphics import plot_decision_boundaries, plot_pca_scatter
from etalon import sklearn_model
from svm import SVM


def custom_model(X_train, X_test, y_train, y_test):

    model = SVM(kernel='linear', C=0.005)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    Metrics.print_all("CUSTOM Model", y_test, y_pred)

    return model.coef_[0], -model.b, model.support_vectors

def main_pipeline(model_pipe):

    df = load_and_prepare_data()

    X = df.drop(columns='target')

    y = df['target']

    X = scale_features(X)

    X = X.to_numpy()

    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    w, b, support_vectors = model_pipe(X_train, X_test, y_train, y_test)

    pca, principal_components = compute_pca(X, n_components=2)

    fig, ax = plot_pca_scatter(principal_components, y)

    plot_decision_boundaries(ax, pca, w, b, 
                            label='Разделяющая прямая', 
                            color='red',
                            support_vectors=support_vectors,
                            X_train=X_train,
                            y_train=y_train)

    ax.legend()


main_pipeline(sklearn_model)

main_pipeline(custom_model)

plt.show()