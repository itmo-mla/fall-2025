# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv('../data/heart.csv')
data.head()

# %%
X = data.drop(columns=['target']).values.astype(np.float32)
y = data['target'].values.astype(np.float32).reshape(-1, 1)
y = [val[0] for val in y]
y = np.array(y)

# %%
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# %% [markdown]
# # Проверка работы алгоритма Ньютона-Рафсона

# %%
from logreg import LogisticRegression

# %%

nr_model = LogisticRegression(max_iter=1000, tol=1e-6)
nr_model.fit_newton_raphson(X_train, y_train)
y_pred = nr_model.predict(X_test)

nr_acc = accuracy_score(y_test, y_pred)
nr_prec = precision_score(y_test, y_pred)
nr_rec = recall_score(y_test, y_pred)
nr_f1 = f1_score(y_test, y_pred)
nr_roc_auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {nr_acc:.4f}")
print(f"Precision: {nr_prec:.4f}")
print(f"Recall: {nr_rec:.4f}")
print(f"F1-score: {nr_f1:.4f}")
print(f"ROC AUC: {nr_roc_auc:.4f}")

# %% [markdown]
# # Проверка работы алгоритма IRLS

# %%
ls_model = LogisticRegression(max_iter=1000, tol=1e-6)
ls_model.fit_irls(X_train, y_train)
y_pred = ls_model.predict(X_test)

ls_acc = accuracy_score(y_test, y_pred)
ls_prec = precision_score(y_test, y_pred)
ls_rec = recall_score(y_test, y_pred)
ls_f1 = f1_score(y_test, y_pred)
ls_roc_auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {ls_acc:.4f}")
print(f"Precision: {ls_prec:.4f}")
print(f"Recall: {ls_rec:.4f}")
print(f"F1-score: {ls_f1:.4f}")
print(f"ROC AUC: {ls_roc_auc:.4f}")

# %%
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

sk_model = SklearnLogisticRegression(max_iter=1000, tol=1e-6)
sk_model.fit(X_train, y_train)
sk_y_pred = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
sk_prec = precision_score(y_test, sk_y_pred)
sk_rec = recall_score(y_test, sk_y_pred)
sk_f1 = f1_score(y_test, sk_y_pred)
sk_roc_auc = roc_auc_score(y_test, sk_y_pred)
print(f"Sklearn Accuracy: {sk_acc:.4f}")
print(f"Sklearn Precision: {sk_prec:.4f}")
print(f"Sklearn Recall: {sk_rec:.4f}")
print(f"Sklearn F1-score: {sk_f1:.4f}")
print(f"Sklearn ROC AUC: {sk_roc_auc:.4f}")

# %%

from pyexpat import model
from numpy.linalg import norm

cosine_similarity = np.dot(nr_model.weights.flatten(), ls_model.weights.flatten()) / (norm(nr_model.weights) * norm(ls_model.weights))
print(f"Cosine similarity between NR and IRLS coefficients: {cosine_similarity:.4f}")   

cosine_similarity = np.dot(nr_model.weights.flatten(), sk_model.coef_.flatten()) / (norm(nr_model.weights) * norm(sk_model.coef_))
print(f"Cosine similarity between NR and Sklearn coefficients: {cosine_similarity:.4f}")   

cosine_similarity = np.dot(ls_model.weights.flatten(), sk_model.coef_.flatten()) / (norm(ls_model.weights) * norm(sk_model.coef_))
print(f"Cosine similarity between IRLS and Sklearn coefficients: {cosine_similarity:.4f}")   


