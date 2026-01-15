# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('../heart.csv')
X = data.drop(columns=['target']).values.astype(np.float32)
Y = data['target'].values.astype(np.float32).reshape(-1, 1)
Y = 2 * Y - 1 

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, shuffle=True
)

# %%
from core import ModelBaseline, ActivationDummy, MSELoss, SGDOptimizer, SteepestGradientDescentOptimizer

# %%
optimizer = SGDOptimizer(lr=1e-6, momentum=0.5, weight_decay=1e-5)
input_dim = X_train.shape[1]

# %%
model = ModelBaseline(
    net_conf=[input_dim, 1],
    activation_func=ActivationDummy(),
    resulting_func=ActivationDummy(),
    loss_func=MSELoss()
)
model.fit(X_train, Y_train, num_epochs=2500, batch_size=1, optim=optimizer)

# %%
model.plot_loss()

# %%
model.margin_plot(X_train, Y_train, red_thresh=-0.46, yellow_thresh=0.46)

# %%
model.evaluate(X_test, Y_test, bin_classifier=True)

# %%
model.evaluate_sequence(X_test, Y_test, alpha=1)

# %% [markdown]
# # Инициализация весов через корреляцию

# %%
def compute_corr_vector(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    corr_vector = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        if np.std(X[:, i]) > 1e-9:  
            corr_vector[i] = np.corrcoef(X[:, i], Y.reshape(-1))[0, 1]
        else:
            corr_vector[i] = 0.0
    return corr_vector

# %%
corr_vector = compute_corr_vector(X_train, Y_train)

# %%
input_dim = X_train.shape[1]
model_corr = ModelBaseline(
    net_conf=[input_dim, 1],
    activation_func=ActivationDummy(),
    resulting_func=ActivationDummy(),
    loss_func=MSELoss(),
    corr_vector=corr_vector
)

# %%
model_corr.fit(X_train, Y_train, num_epochs=2500, batch_size=1, optim=optimizer)

# %%
model_corr.plot_loss()

# %%
model_corr.margin_plot(X_train, Y_train, red_thresh=-0.46, yellow_thresh=0.46)

# %%
model_corr.evaluate(X_test, Y_test, bin_classifier=True)

# %%
model_corr.evaluate_sequence(X_test, Y_test, alpha=1)

# %% [markdown]
# # Обучение модели через мультистарт

# %%
best_Q = np.inf
model_multi = None

for starts in range(20):
    curr_model = ModelBaseline(net_conf=[input_dim, 1],
                               activation_func=ActivationDummy(),
                               resulting_func=ActivationDummy(),
                               loss_func=MSELoss())
    
    curr_model.fit(X_train, Y_train, num_epochs=2500, batch_size=1, optim=optimizer)
    curr_Q = curr_model.evaluate_sequence(X_test, Y_test, alpha=1)

    if best_Q > curr_Q:
        best_Q = curr_Q
        model_multi = curr_model
    

# %%
model_multi.plot_loss()

# %%
model_multi.margin_plot(X_train, Y_train, red_thresh=-0.531, yellow_thresh=0.47)

# %%
model_multi.evaluate(X_test, Y_test, bin_classifier=True)

# %%
model_multi.evaluate_sequence(X_test, Y_test, alpha=1)

# %% [markdown]
# # Обучение со случайным предъявлением

# %%
model_rand = ModelBaseline(net_conf=[input_dim, 1],
                               activation_func=ActivationDummy(),
                               resulting_func=ActivationDummy(),
                               loss_func=MSELoss())


model_rand.fit(X_train, Y_train, num_epochs=2500, batch_size=1, optim=optimizer, sampling='random')

# %%
model_rand.plot_loss()

# %%
model_rand.margin_plot(X_train, Y_train, red_thresh=-0.47, yellow_thresh=0.45)

# %%
model_rand.evaluate(X_test, Y_test, bin_classifier=True)

# %%
model_rand.evaluate_sequence(X_test, Y_test, alpha=1)

# %% [markdown]
# # Обучение с предъявлением по модулю отступа

# %%
model_margin = ModelBaseline(net_conf=[input_dim, 1],
                               activation_func=ActivationDummy(),
                               resulting_func=ActivationDummy(),
                               loss_func=MSELoss())


model_margin.fit(X_train, Y_train, num_epochs=2500, batch_size=1, optim=optimizer, sampling='margin')

# %%
model_margin.plot_loss()

# %%
model_margin.margin_plot(X_train, Y_train, red_thresh=-0.65, yellow_thresh=0.6)

# %%
model_margin.evaluate(X_test, Y_test, bin_classifier=True)

# %%
model_margin.evaluate_sequence(X_test, Y_test, alpha=1)

# %% [markdown]
# # Эталонная реализация (Scikit-Learn)

# %%
from sklearn.linear_model import SGDClassifier


Y_train_sk = np.array([el.item() if el.item() == 1 else -1 for el in Y_train])
Y_test_sk = np.array([el.item() if el.item() == 1 else -1 for el in Y_test])

# optimizer = SGDOptimizer(lr=5 * 1e-8, momentum=0.1, weight_decay=1e-4)
# model_corr.fit(X_train, Y_train, num_epochs=250, batch_size=1, optim=optimizer)

model_ref = SGDClassifier(loss='squared_error',
                          penalty='l2',
                          learning_rate='constant',
                          eta0=1e-6,
                          max_iter=10000,
                          tol=None,
                          verbose=1)


model_ref.fit(X_train, Y_train_sk)


Y_pred = model_ref.predict(X_test)

Y_pred_sk = np.array([1 if el.item() >= 0 else -1 for el in Y_pred])

# %%
intersec = np.array((Y_pred_sk == Y_test_sk))
accuracy = len(intersec[intersec == True]) / len(Y_test_sk)
accuracy

# %% [markdown]
# # Переход к MAE loss

# %%
from core import MAELoss

# %%
model_mae = ModelBaseline(net_conf=[input_dim, 1],
                               activation_func=ActivationDummy(),
                               resulting_func=ActivationDummy(),
                               loss_func=MAELoss())

# %%
optim_mae = SGDOptimizer(lr=1e-6,
                         momentum=0.5,
                         weight_decay=1e-5)
model_mae.fit(X_train, Y_train, num_epochs=5000, batch_size=1, optim=optim_mae)

# %%
model_mae.plot_loss()

# %%
model_mae.margin_plot(X_train, Y_train, red_thresh=-0.98, yellow_thresh=0.82)

# %%
model_mae.evaluate(X_test, Y_test, True)

# %%
model_mae.evaluate_sequence(X_test, Y_test, alpha=1)

# %%
model_ref_mae = SGDClassifier(loss='epsilon_insensitive',
                          penalty='l2',
                          learning_rate='constant',
                          eta0=1e-6,
                          max_iter=30000,
                          tol=None,
                          verbose=1,
                          epsilon=0)

model_ref_mae.fit(X_train, Y_train_sk)

Y_pred_mae = model_ref_mae.predict(X_test)
Y_pred_sk_mae = np.array([1 if el.item() >= 0 else -1 for el in Y_pred_mae])

# %%
intersec_mae = np.array((Y_pred_sk_mae == Y_test_sk))
accuracy_mae = len(intersec_mae[intersec_mae == True]) / len(Y_test_sk)
accuracy_mae


