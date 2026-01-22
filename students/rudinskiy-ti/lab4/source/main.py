from PCA import HandmadePCA
from data_preprocess import prepare_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def draw_plot(Em):
  x = [i for i in range(len(Em))]
  y = Em
  plt.plot(x, Em)
  plt.plot([5, 5], [0, Em[5]], 'r--', linewidth=2, label=f'x = {5}')
  plt.plot([x[0], x[-1]], [y[0], y[-1]], 'r--', linewidth=1, color='grey')
  plt.scatter(x, y)
  plt.fill_between(x, y, color='lightblue', alpha=0.5, label='Площадь под графиком')
  plt.show()

df = prepare_dataset()
X = df.drop('annual_income_usd', axis='columns')
X_sk = X.copy()
m1 = HandmadePCA(True)
Em = m1.fit(X)
X_tr = m1.transform(X)
m2 = PCA(m1.n_components)
m2.fit(X_sk)
X_sk_tr = m2.transform(X_sk)
print(f"{np.linalg.norm(np.abs(X_tr) - np.abs(X_sk_tr)):.6e}")
print(m1.explained_variance_ratio - m2.explained_variance_ratio_)
draw_plot(Em)