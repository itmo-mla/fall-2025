from loss.LogLoss import LogLoss
from optimizer.momentum import Momentum
import numpy as np




class SGDClassifier:
    def __init__(self, learning_rate=0.01, alpha=1e-4, n_iterations=1000,
                 optimizer=Momentum(), loss=LogLoss(), weight_init='random',
                 penalty=None, lambd=0.5, ordering='random', use_bias=True):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.loss = loss
        self.penalty = penalty
        self.Q = 0
        self.lambd = lambd
        self.ordering = ordering
        self.losses = []
        self.use_bias = use_bias
        np.random.seed(42)

    def _add_bias_column(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _initialize_weights(self, n_features, y=None, X=None):
        match self.weight_init:
            case 'random':
                self.weights = (1/n_features)*np.random.random_sample(n_features) - 1/(2*n_features)
                if self.use_bias:
                    self.weights[-1] = 0.0 
            case 'correlation':
            
                w = np.array([np.dot(X[:, i], y) / np.dot(X[:, i], X[:, i]) for i in range(n_features - (1 if self.use_bias else 0))])
                if self.use_bias:
                    w = np.append(w, 0.0)
                self.weights = w
            case 'multi':
                a = [SGDClassifier(learning_rate=self.learning_rate, alpha=self.alpha,
                                    n_iterations=5, optimizer=self.optimizer, loss=self.loss,
                                    weight_init='random', penalty=self.penalty, use_bias=self.use_bias) for _ in range(10)]
                X_ = X[:, :-1]
                _ = [i.fit(X_, y) for i in a]
                best = min(a, key=lambda model: model.Q)
                self.weights = best.weights
                print(self.weights.shape)

    def fit(self, X, y):
       

        X = self._add_bias_column(X) if self.use_bias else X
        n_samples, n_features = X.shape
        self._initialize_weights(n_features, y, X)

        print(X.shape)

        self.Q = self.loss.loss(y * (X @ self.weights)).mean()

        for iter in range(self.n_iterations):
            match self.ordering:
                case 'random':
                    shuffle_indices = np.random.permutation(n_samples)
                case 'margin-first':
                    margin=np.abs(y * (X @ self.weights))
                    shuffle_indices = (margin).argsort()
                    print()

            for _x, _y in zip(X[shuffle_indices], y[shuffle_indices]):
                match self.penalty:
                    case 'l2':
                        def grad_fun(w):
                            reg = self.alpha * w
                            return self.loss.derivative(_y * (_x @ w)) * _y * _x + reg

                        updated_weights = self.optimizer.update(self.weights, grad_fun, self.learning_rate)
                        self.weights = updated_weights

                    case None:
                        def grad_fun(w):
                            return self.loss.derivative(_y * (_x @ w)) * _y * _x
                        updated_weights = self.optimizer.update(self.weights, grad_fun, self.learning_rate)
                        self.weights = updated_weights

                    case _:
                        raise ValueError("penalty must be 'l2' or None")

                loss = self.loss.loss(y * (X @ self.weights)).mean()

                updated_q = self.lambd * self.loss.loss(_y * (_x @ self.weights)) + (1 - self.lambd) * self.Q
                self.Q = updated_q

            self.losses.append(loss)
            if iter % 10 == 0:
                print(f"Iter {iter}: loss={loss:.4f}, |w|={np.linalg.norm(self.weights):.4f}")

    def predict(self, X):
        if self.use_bias:
            linear_model = (X @ self.weights[:-1]) + self.weights[-1]
        else:
            linear_model = X @ self.weights
        return np.where(linear_model >= 0, 1, -1)
          
