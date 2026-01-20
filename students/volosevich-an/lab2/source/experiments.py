# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('../data/heart.csv')
X = data.drop(columns=['target']).values.astype(np.float32)
y = data['target'].values.astype(np.float32).reshape(-1, 1)
y = 2 * y - 1 
y = [val[0] for val in y]
y = np.array(y)

# %%
from knn import KNNClassifier
from proto import PrototypeSelector
from utils import plot_empirical_risk_k, plot_empirical_risk_h, visualize_prototypes

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
model = KNNClassifier(k=5, window='fixed', h=11, p=2)
model.fit(X_train, y_train)
print(f'Train score: {model.score(X_train, y_train)}\n'+
      f'Test score: {model.score(X_test, y_test)}')

# %%
plot_empirical_risk_h(model, X_train, y_train, X_test, y_test, h_values=range(2, 30))

# %%
model = KNNClassifier(k=5, window='dynamic', p=2)
model.fit(X_train, y_train)
print(f'Train score: {model.score(X_train, y_train)}\n'+
      f'Test score: {model.score(X_test, y_test)}')

# %%
from sklearn.neighbors import KNeighborsClassifier
model_ref = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2, algorithm='brute')
model_ref.fit(X_train, y_train)
print(f'Train score: {model.score(X_train, y_train)}\n'+
      f'Test score: {model.score(X_test, y_test)}')

# %%
best_k = model.loo_estimate_k(k_values=list(range(1, 100)))
best_k

# %%
model = KNNClassifier(k=best_k[0], window='dynamic', h=1.0, p=2)
model.fit(X_train, y_train)
print(f'Train score: {model.score(X_train, y_train)}\n'+
      f'Test score: {model.score(X_test, y_test)}')

# %%
plot_empirical_risk_k(model, X_train, y_train, X_test, y_test ,k_values=range(2, 30))

# %%
selector = PrototypeSelector(n_centroids=1, n_border=2)
X_new, y_new = selector.select(X_train, y_train)
visualize_prototypes(X_train, y_train, X_new, y_new)

# %%
model.fit(X_new, y_new)
print(f'Train score: {model.score(X, y)}\n'+
      f'Test score: {model.score(X_test, y_test)}')


