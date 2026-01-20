# %%
import pandas as pd

# %%
df = pd.read_csv('../data/rent.csv')
df

# %% [markdown]
# # Data Preprocessing

# %%
df.info()

# %%
df.drop('house_type', axis=1, inplace=True)
df

# %%
import numpy as np

print(len(np.unique(df['locality'])))
print(np.unique(df['city']))

# %%
df.drop('locality', axis=1, inplace=True)
df.drop('city', axis=1, inplace=True)
df.drop('furnishing', axis=1, inplace=True)
df

# %%
new_df = pd.get_dummies(df, drop_first=False)
new_df

# %% [markdown]
# # Using linear regression as a validation baseline

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model = LinearRegression()

y = new_df.pop('rent').to_numpy()
X = new_df.to_numpy()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# %%
model.fit(X_train, y_train)

print('Train Accuracy:', model.score(X_train, y_train))
print('Test Accuracy:', model.score(X_test, y_test))

# %% [markdown]
# # Using PCA for simpler data set

# %%
from sklearn.decomposition import PCA as PCA_SK

pca_sk = PCA_SK(n_components=0.95)

X_reduced_sk = pca_sk.fit_transform(X)
X_reduced_sk
X_train, X_test, y_train, y_test = train_test_split(X_reduced_sk, y, test_size=0.2, random_state=21)
model.fit(X_train, y_train)
print('Effective dimension (sklearn PCA):', pca_sk.n_components_)
print('Train Accuracy after PCA:', model.score(X_train, y_train))
print('Test Accuracy after PCA:', model.score(X_test, y_test))   

# %%
from pca import PCA

pca = PCA()
X_reduced = pca.fit_transform(X)
X_reduced
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=21)
model.fit(X_train, y_train)
print('Train Accuracy after PCA:', model.score(X_train, y_train))
print('Test Accuracy after PCA:', model.score(X_test, y_test))   

# %%
from visualize import plot_explained_variance, plot_pca_scatter, plot_pca_3d_scatter


plot_explained_variance(pca, X)

# %%
plot_pca_scatter(pca, X_reduced, y)

# %%
plot_pca_scatter(pca_sk, X_reduced_sk, y)


