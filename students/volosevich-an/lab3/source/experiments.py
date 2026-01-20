# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('../data/heart.csv')
X = data.drop(columns=['target']).values.astype(np.float32)
X = ((X - np.min(X)) / (np.max(X) - np.min(X)))
y = data['target'].values.astype(np.float32).reshape(-1, 1)
y = 2 * y - 1 
y = [val[0] for val in y]
y = np.array(y)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
from svm_dual import SVMDual
from visualize import plot_svm_pca

# %%
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

# %% [markdown]
# ## Эталон

# %%
from sklearn.svm import SVC
model_sklearn = SVC(C=1.0, kernel='linear', gamma=1.0)
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)
print(f'Sklearn Train score: {accuracy(y, y_pred_sklearn)}')

# %%
model_sklearn = SVC(C=1.0, kernel='poly', coef0=1, gamma=1.0)
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)
print(f'Sklearn Train score: {accuracy(y, y_pred_sklearn)}')

# %%
model_sklearn = SVC(C=1.0, kernel='rbf', gamma=1.0)
model_sklearn.fit(X, y)
y_pred_sklearn = model_sklearn.predict(X)
print(f'Sklearn Train score: {accuracy(y, y_pred_sklearn)}')

# %% [markdown]
# ## Собственная реализация

# %%
model_linear = SVMDual(C=1.0, kernel='linear', gamma=1.0, verbose=True)
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)
print(f'Train score: {accuracy(y, y_pred_linear)}')

# %%
plot_svm_pca(model_linear, X, y)

# %%
model_poly = SVMDual(C=1.0, kernel='poly', gamma=1.0, coef0=1, verbose=True)
model_poly.fit(X, y)
y_pred_poly = model_poly.predict(X)
print(f'Train score: {accuracy(y, y_pred_poly)}')

# %%
plot_svm_pca(model_poly, X, y)

# %%
model_rbf = SVMDual(C=1.0, kernel='rbf', gamma=1.0, verbose=True)
model_rbf.fit(X, y)
y_pred_rbf = model_rbf.predict(X)
print(f'Train score: {accuracy(y, y_pred_rbf)}')

# %%
plot_svm_pca(model_rbf, X, y)


