import matplotlib.pyplot as plt

from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
from etalon import sklearn_model
from custom_model import model_workflow
from pca import compute_pca
from graphics import plot_decision_boundaries, plot_pca_scatter

df = load_and_prepare_data()

X = df.drop(columns='target')

y = df['target']

X = scale_features(X)

pca, principal_components = compute_pca(X, n_components=2)

X_train, X_test, y_train, y_test = train_test_split_data(X, y)

etalon_w, etalon_b = sklearn_model(X_train, X_test, y_train, y_test)

custom_w, custom_b = model_workflow("Default Model", X_train, X_test, y_train, y_test)

fig, ax = plot_pca_scatter(principal_components, y)

plot_decision_boundaries(ax, pca, etalon_w, etalon_b, label='Разделяющая прямая sklearn', color='red')

plot_decision_boundaries(ax, pca, custom_w, custom_b, label='Разделяющая прямая custom', color='green')

ax.legend()

plt.show()