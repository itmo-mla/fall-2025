import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


class LogisticRegressionCustom:
    def __init__(self, max_iter=1000, tol=1e-8, C=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
    
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def sigmoid(self, z):
        return expit(z)
    
    def loss(self, y, y_pred, weights, X_aug):
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        log_loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        
        # в sklearn используется l2регуляризация мы тоже будем
        if self.C != np.inf:
            reg_term = 0.5 * np.sum(weights[1:]**2) / self.C
            return log_loss + reg_term / X_aug.shape[0]
        return log_loss
    
    def fit_newton(self, X, y):
        """
        Метод Ньютона–Рафсона для логистической регрессии.
        На каждой итерации считаем градиент и Гессиан, делаем шаг:
        beta_{t+1} = beta_t - H^-1 * grad
        """
        X_aug = self._add_intercept(X)  # добавляем колонку единиц для свободного члена
        
        n_samples, n_features = X_aug.shape
        weights = np.zeros(n_features)  # начальное приближение
        prev_loss = float('inf')
        
        for i in range(self.max_iter):
            # 1. Линейное предсказание
            linear_pred = X_aug @ weights
            # 2. Сигмоида для вероятностей
            y_pred = self.sigmoid(linear_pred)
            
            # 3. Градиент log-loss (с учётом регуляризации)
            gradient = X_aug.T @ (y_pred - y) / n_samples
            if self.C != np.inf:
                reg_grad = weights.copy()
                gradient += reg_grad / (self.C * n_samples)
            
            # 4. Гессиан: W = diag(p*(1-p))
            W = y_pred * (1 - y_pred)
            hessian = (X_aug.T * W) @ X_aug / n_samples
            if self.C != np.inf:
                reg_matrix = np.eye(n_features) / (self.C * n_samples)
                reg_matrix[0, 0] = 0  # не регуляризуем свободный член
                hessian += reg_matrix
            
            # 5. Шаг Ньютона
            weights -= np.linalg.solve(hessian, gradient)
            
            # 6. Сохраняем loss для анализа сходимости
            current_loss = self.loss(y, y_pred, weights, X_aug)
            self.loss_history.append(current_loss)
            
            # 7. Проверка сходимости
            if i > 0 and abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss
        
        self.intercept_ = weights[0]
        self.coef_ = weights[1:].reshape(1, -1)
        return self

    
    def fit_irls(self, X, y):
        """
        Метод IRLS для логистической регрессии.
        На каждой итерации делаем взвешенную линейную регрессию:
        1. Вычисляем σ_i = сигмоида
        2. Вычисляем веса γ_i = p*(1-p)
        3. Вычисляем псевдо-отклик z
        4. Решаем взвешенную линейную регрессию для обновления w
        """
        X_aug = self._add_intercept(X)
        n_samples, n_features = X_aug.shape
        weights = np.zeros(n_features)
        prev_loss = float('inf')
        
        for i in range(self.max_iter):
            # 1. Линейное предсказание
            linear_pred = X_aug @ weights
            # 2. Сигмоида для вероятностей
            y_pred = self.sigmoid(linear_pred)
            
            # 3. Вычисляем веса γ_i = p*(1-p)
            W = y_pred * (1 - y_pred)
            # 4. Вычисляем псевдо-отклик z = linear_pred + (y - y_pred)/W
            z = linear_pred + (y - y_pred) / (W + 1e-10)
            
            # 5. Взвешенная матрица для решения МНК
            W_matrix = np.diag(W)
            XTWX = X_aug.T @ W_matrix @ X_aug
            XTWz = X_aug.T @ W_matrix @ z
            
            # 6. Добавляем регуляризацию
            if self.C != np.inf:
                reg_matrix = np.eye(n_features) / self.C
                reg_matrix[0, 0] = 0  # свободный член не регуляризуем
                XTWX += reg_matrix
            
            # 7. Решаем систему для нового шага
            weights = np.linalg.solve(XTWX, XTWz)
            
            # 8. Сохраняем log-loss для анализа сходимости
            current_loss = self.loss(y, y_pred, weights, X_aug)
            self.loss_history.append(current_loss)
            
            # 9. Проверка сходимости
            if i > 0 and abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss
        
        # 10. Сохраняем результаты
        self.intercept_ = weights[0]
        self.coef_ = weights[1:].reshape(1, -1)
        return self

    
    def predict_proba(self, X):
        X_aug = self._add_intercept(X) 
        
        weights = np.concatenate([[self.intercept_], self.coef_[0]]) 
        
        linear_pred = X_aug @ weights
        probabilities = self.sigmoid(linear_pred)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

C_value = 10 


newton_model = LogisticRegressionCustom(max_iter=1000, tol=1e-8, C=C_value)
newton_model.fit_newton(X_train, y_train)

irls_model = LogisticRegressionCustom(max_iter=1000, tol=1e-8, C=C_value)
irls_model.fit_irls(X_train, y_train)

sklearn_model = LogisticRegression(
    penalty='l2',
    C=C_value,
    solver='lbfgs',
    max_iter=1000,
    tol=1e-8,
    fit_intercept=True,
    random_state=42
)
sklearn_model.fit(X_train, y_train)

plt.figure(figsize=(8, 5))

plt.plot(
    newton_model.loss_history,
    label="Newton–Raphson",
    linewidth=2
)

plt.plot(
    irls_model.loss_history,
    label="IRLS",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Log-loss")
plt.title(f"Сходимость логистической регрессии (C = {C_value})")
plt.legend()
plt.grid(True)

plt.show()


print("\nРАЗНИЦА В ВЕСАХ:")
newton_weights = np.concatenate([[newton_model.intercept_], newton_model.coef_[0]])
irls_weights = np.concatenate([[irls_model.intercept_], irls_model.coef_[0]])
sklearn_weights = np.concatenate([sklearn_model.intercept_, sklearn_model.coef_[0]])

print(f"Ньютон vs IRLS:    max={np.max(np.abs(newton_weights - irls_weights)):.6f}")
print(f"Ньютон vs Sklearn: max={np.max(np.abs(newton_weights - sklearn_weights)):.6f}")
print(f"IRLS vs Sklearn:   max={np.max(np.abs(irls_weights - sklearn_weights)):.6f}")

# Проверка точности на теста
test_acc_newton = np.mean(newton_model.predict(X_test) == y_test)
test_acc_irls = np.mean(irls_model.predict(X_test) == y_test)
test_acc_sklearn = np.mean(sklearn_model.predict(X_test) == y_test)

print(f"ТОЧНОСТЬ на тесте:    Ньютон={test_acc_newton:.4f}, IRLS={test_acc_irls:.4f}, Sklearn={test_acc_sklearn:.4f}")

# Проверка вероятностей
y_proba_newton = newton_model.predict_proba(X_test)[:, 1]
y_proba_irls = irls_model.predict_proba(X_test)[:, 1]
y_proba_sklearn = sklearn_model.predict_proba(X_test)[:, 1]

print(f"\nРазница в вероятностях:")
print(f"Ньютон vs IRLS:    {np.max(np.abs(y_proba_newton - y_proba_irls)):.6f}")
print(f"Ньютон vs Sklearn: {np.max(np.abs(y_proba_newton - y_proba_sklearn)):.6f}")
