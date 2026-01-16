import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
from vizualization import visualize_svm

X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nИНФОРМАЦИЯ О БАЛАНСЕ КЛАССОВ")
print(f"Количество примеров класса -1 в тренировочной выборке: {np.sum(y_train == 0)}")
print(f"Количество примеров класса +1 в тренировочной выборке: {np.sum(y_train == 1)}")
print(f"Количество примеров класса -1 в тестовой выборке: {np.sum(y_test == 0)}")
print(f"Количество примеров класса +1 в тестовой выборке: {np.sum(y_test == 1)}")
print()

sklearn_svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
sklearn_svm_rbf.fit(X_train, y_train)
y_pred_rbf = sklearn_svm_rbf.predict(X_test)

sklearn_svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
sklearn_svm_linear.fit(X_train, y_train)
y_pred_linear = sklearn_svm_linear.predict(X_test)

print(f"RBF ядро sklearn:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")
print(f"  f1-Score: {f1_score(y_test, y_pred_rbf):.4f}")
print(f"  Опорных векторов: {len(sklearn_svm_rbf.support_vectors_)}")

print(f"\n линейное ядро sklearn:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(f"  1-Score: {f1_score(y_test, y_pred_linear):.4f}")
print(f"  Опорных векторов: {len(sklearn_svm_linear.support_vectors_)}")

class CustomRBFSVM:
    def __init__(self, C=1.0, gamma='scale', max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter

        self.X_train = None
        self.y_train = None
        self.lambdas = None
        self.w0 = None
        self.gamma_val = None
        self.support_vectors_ = None
        self.support_indices_ = None

    def rbf_kernel_custom(self, X1, X2, gamma):
        X1_norm = np.sum(X1 ** 2, axis=1)
        X2_norm = np.sum(X2 ** 2, axis=1)
        pairwise_dots = X1 @ X2.T
        distances_sq = X1_norm[:, None] + X2_norm[None, :] - 2 * pairwise_dots
        return np.exp(-gamma * distances_sq)

    def prepare_dual_problem_rbf(self, X, y):
        self.gamma_val = 1.0 / (X.shape[1] * X.var()) if X.shape[1] > 0 else 1.0
        K = self.rbf_kernel_custom(X, X, self.gamma_val)
        Q = np.outer(y, y) * K
        bounds = [(0, self.C) for _ in range(X.shape[0])]
        return Q, K, bounds

    def solve_dual_problem(self, Q, y, bounds):
        n = len(y)
        lambdas0 = np.zeros(n)

        def objective_func(lambdas):
            return 0.5 * lambdas @ Q @ lambdas - lambdas.sum()

        def gradient_func(lambdas):
            return Q @ lambdas - np.ones_like(lambdas)

        def constraint_func(lambdas):
            return y @ lambdas

        constraints = {'type': 'eq', 'fun': constraint_func}

        result = minimize(
            fun=objective_func,
            x0=lambdas0,
            method='SLSQP',
            jac=gradient_func,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': 1e-6, 'disp': False}
        )
        lambdas_opt = result.x
        lambdas_opt[lambdas_opt < 1e-10] = 0
        return lambdas_opt, result

    def compute_w0_rbf(self, lambdas, y, K_train):
        mask = (lambdas > 1e-9) & (lambdas < self.C - 1e-9)
        if not mask.any():
            mask = lambdas > 1e-9
        if mask.any():
            indices = np.where(mask)[0]
            w0_values = []
            for i in indices:
                decision = np.dot(K_train[:, i], lambdas * y)
                w0 = y[i] - decision
                w0_values.append(w0)
            return np.mean(w0_values)
        return 0.0

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = np.where(y == 0, -1, 1)
        Q, K_train, bounds = self.prepare_dual_problem_rbf(X, self.y_train)
        self.lambdas, result = self.solve_dual_problem(Q, self.y_train, bounds)
        self.w0 = self.compute_w0_rbf(self.lambdas, self.y_train, K_train)
        self.support_indices_ = np.where(self.lambdas > 1e-8)[0]
        self.support_vectors_ = X[self.support_indices_]
        return self

    def predict(self, X):
        K_test = self.rbf_kernel_custom(X, self.X_train, self.gamma_val)
        decisions = K_test @ (self.lambdas * self.y_train) + self.w0
        predictions = np.where(decisions >= 0, 1, -1)
        return np.where(predictions == -1, 0, 1)

    def decision_function(self, X):
        K_test = self.rbf_kernel_custom(X, self.X_train, self.gamma_val)
        return K_test @ (self.lambdas * self.y_train) + self.w0

our_svm = CustomRBFSVM(C=1.0, gamma='scale', max_iter=1000)
our_svm.fit(X_train, y_train)

y_pred_our = our_svm.predict(X_test)
accuracy_our = accuracy_score(y_test, y_pred_our)
f1_our = f1_score(y_test, y_pred_our)

print(f"\nRBF самописный:")
print(f"  Accuracy: {accuracy_our:.4f}")
print(f"  F1-Score: {f1_our:.4f}")
print(f"  Опорных векторов: {len(our_svm.support_vectors_)}")

visualize_svm(X_test, y_test, sklearn_svm_linear, title="Linear SVM sklearn")
visualize_svm(X_test, y_test, sklearn_svm_rbf, title="RBF SVM sklearn")
visualize_svm(X_test, y_test, our_svm, title="my RBF")
