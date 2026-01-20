import numpy as np
from typing import Tuple, Optional


class LinearClassifier:
    # Линейный классификатор с методом стохастического градиентного спуска.
    
    def __init__(self, n_features: int, learning_rate: float = 0.01,
                 reg_coef: float = 0.001, momentum: float = 0.9,
                 random_state: int = 42, loss_type: str = 'logistic'):

        # Инициализация линейного классификатора.
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reg_coef = reg_coef
        self.momentum = momentum
        self.random_state = random_state
        self.loss_type = loss_type
        
        # Инициализация весов и смещения
        np.random.seed(random_state)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0
        
        # История для отслеживания
        self.loss_history = []
        self.accuracy_history = []
        self.margin_history = []
    
    def margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Вычисление отступов для объектов.
        scores = X @ self.weights + self.bias
        margins = y * scores
        return margins
    
    def loss_function(self, margin: float) -> float:
        # Функция потерь в зависимости от отступа.
        if self.loss_type == 'logistic':
            return np.log(1 + np.exp(-margin))
        elif self.loss_type == 'quadratic':
            return max(0, 1 - margin) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def loss_gradient(self, margin: float) -> float:
        # Градиент функции потерь по отступу.
        if self.loss_type == 'logistic':
            return -1 / (1 + np.exp(margin))
        elif self.loss_type == 'quadratic':
            if margin >= 1:
                return 0
            else:
                return -2 * (1 - margin)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray, 
                     include_reg: bool = True) -> float:
        # Вычисление эмпирического риска с L2 регуляризацией.
        margins = self.margin(X, y)
        data_loss = np.mean([self.loss_function(m) for m in margins])
        
        if include_reg:
            reg_loss = 0.5 * self.reg_coef * np.sum(self.weights ** 2)
            return data_loss + reg_loss
        return data_loss
    
    def compute_gradient(self, X_i: np.ndarray, y_i: float) -> Tuple[np.ndarray, float]:
        # Вычисление градиента функции потерь для одного объекта.
        margin = y_i * (np.dot(self.weights, X_i) + self.bias)
        
        dL_dM = self.loss_gradient(margin)

        grad_w = dL_dM * y_i * X_i

        grad_b = dL_dM * y_i

        grad_norm = np.linalg.norm(grad_w)
        if grad_norm > 1.0:
            grad_w = grad_w / grad_norm
            grad_b = grad_b / grad_norm
        
        return grad_w, grad_b
    
    def adaptive_learning_rate(self, X_i: np.ndarray, epoch: int) -> float:
 
        # Затухание learning rate со временем
        base_lr = self.learning_rate / (1 + 0.01 * epoch)
        
        # Для квадратичной функции потерь оптимальный шаг h = 1/||x_i||^2
        if self.loss_type == 'quadratic':
            norm_sq = np.dot(X_i, X_i)
            if norm_sq > 1e-10:
                return min(base_lr, 1.0 / norm_sq)
        
        return base_lr
    
    def update_with_momentum(self, grad_w: np.ndarray, grad_b: float, 
                        h: Optional[float] = None):
        if h is None:
            h = self.learning_rate

        self.velocity_w = self.momentum * self.velocity_w + h * grad_w
        self.velocity_b = self.momentum * self.velocity_b + h * grad_b

        self.weights -= self.velocity_w + h * self.reg_coef * self.weights
        self.bias -= self.velocity_b

    def recursive_loss_estimate(self, current_loss: float, 
                               new_loss: float, lambda_forget: float = 0.1) -> float:
        
        # Рекуррентная оценка функционала качества через экспоненциальное скользящее среднее.
        return lambda_forget * new_loss + (1 - lambda_forget) * current_loss
    
    def select_by_margin(self, X: np.ndarray, y: np.ndarray, 
                        n_select: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        # Выбор объектов для обучения по модулю отступа (наименьшие |M_i|).
        margins = np.abs(self.margin(X, y))
        indices = np.argsort(margins)[:n_select]
        return X[indices], y[indices]
    
    def fit_sgd(self, X: np.ndarray, y: np.ndarray, 
               n_epochs: int = 100, batch_size: int = 1,
               adaptive_lr: bool = False,
               margin_selection: bool = False,
               verbose: bool = True) -> dict:
      
        n_samples = X.shape[0]
        history = {
            'loss': [],
            'accuracy': [],
            'avg_margin': [],
            'weights_norm': []
        }
        
        # Инициализация оценки функционала
        Q_estimate = self.compute_loss(X, y)
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                if margin_selection and i == 0:
                    # В начале каждой эпохи выбираем сложные объекты
                    X_batch, y_batch = self.select_by_margin(
                        X_shuffled, y_shuffled, batch_size
                    )
                else:
                    end_idx = min(i + batch_size, n_samples)
                    X_batch = X_shuffled[i:end_idx]
                    y_batch = y_shuffled[i:end_idx]
                
                # Вычисление градиента и обновление для каждого объекта в батче
                for X_i, y_i in zip(X_batch, y_batch):

                    grad_w, grad_b = self.compute_gradient(X_i, y_i)
                    
                    if adaptive_lr:
                        h = self.adaptive_learning_rate(X_i, epoch)
                    else:
                        h = self.learning_rate / (1 + 0.01 * epoch)
                    
                    self.update_with_momentum(grad_w, grad_b, h)
                    
                    # Рекуррентная оценка функционала
                    loss_i = self.loss_function(y_i * (np.dot(self.weights, X_i) + self.bias))
                    Q_estimate = self.recursive_loss_estimate(Q_estimate, loss_i)
                    epoch_losses.append(loss_i)
            
            # Оценка после эпохи
            epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
            accuracy = self.accuracy(X, y)
            avg_margin = np.mean(np.abs(self.margin(X, y)))

            history['loss'].append(epoch_loss)
            history['accuracy'].append(accuracy)
            history['avg_margin'].append(avg_margin)
            history['weights_norm'].append(np.linalg.norm(self.weights))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch+1}/{n_epochs}, "
                      f"Потери: {epoch_loss:.4f}, "
                      f"Точность: {accuracy:.4f}, "
                      f"Средний |M|: {avg_margin:.4f}, "
                      f"||w||: {np.linalg.norm(self.weights):.4f}")
        
        self.loss_history = history['loss']
        self.accuracy_history = history['accuracy']
        self.margin_history = history['avg_margin']
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Предсказание меток классов.
        scores = X @ self.weights + self.bias
        return np.sign(scores)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.loss_type != 'logistic':
            raise ValueError("Probability predictions only available for logistic loss")
        
        scores = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-scores))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def initialize_by_correlation(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        for j in range(n_features):
            f_j = X[:, j]
            numerator = np.dot(y, f_j)
            denominator = np.dot(f_j, f_j)
            
            if abs(denominator) > 1e-10:
                self.weights[j] = numerator / denominator
            else:
                self.weights[j] = 0.0

        self.bias = 0.0

        self.velocity_w = np.zeros(n_features)
        self.velocity_b = 0.0
    
    def reset_weights(self):
        # Сброс весов к случайным значениям
        np.random.seed(self.random_state)
        self.weights = np.random.randn(self.n_features) * 0.01
        self.bias = 0.0
        self.velocity_w = np.zeros(self.n_features)
        self.velocity_b = 0.0