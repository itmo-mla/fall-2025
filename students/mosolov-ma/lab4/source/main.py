from pca import custom_pca, CustomPCA
from data_workflow import load_and_prepare_data, scale_features, generate_multicollinear_data
from sklearn.decomposition import PCA
from graphics import plot_pca_scatter, plot_Em
import matplotlib.pyplot as plt
import numpy as np

df = load_and_prepare_data()

X = df.drop(columns='target')

y = df['target']

X = scale_features(X)

X = X.to_numpy()

y = y.to_numpy()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_custom = CustomPCA(n_components=2)
X_pca_custom = pca_custom.fit_transform(X)

plot_pca_scatter(X_pca, y, title='Sklearn PCA')

plot_pca_scatter(X_pca_custom, y, title='custom PCA')

plt.show()

print(pca.singular_values_)

print(pca_custom.singular_values_)

X_synthetic = generate_multicollinear_data(n_samples=1000, noise_std=0.5)

pca = PCA()

X_pca = pca.fit_transform(X_synthetic)

pca_custom = CustomPCA()

X_pca_custom = pca_custom.fit_transform(X_synthetic)

print(pca.singular_values_)

print(pca_custom.singular_values_)

plot_Em(pca.singular_values_, title='Sklearn PCA')

plot_Em(pca_custom.singular_values_, title='custom PCA')

plt.show()