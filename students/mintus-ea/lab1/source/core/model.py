import numpy as np
from .loss import QuadraticLoss

class LinearClassifier:
    def __init__(self, 
                 learning_rate=0.01, 
                 n_epochs=100, 
                 batch_size=1, 
                 momentum=0.9, 
                 reg_alpha=0.01,
                 init_strategy='random',
                 presentation_strategy='random',
                 use_nesterov=False):
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.reg_alpha = reg_alpha  # L2 regularization coefficient
        self.init_strategy = init_strategy
        self.presentation_strategy = presentation_strategy
        self.use_nesterov = use_nesterov
        
        self.w = None
        self.loss_fn = QuadraticLoss()
        self.history = {'loss': [], 'margin_distribution': []}

    def _initialize_weights(self, X, y):
        n_features = X.shape[1]
        
        if self.init_strategy == 'random':
            self.w = np.random.randn(n_features) * 0.01
            
        elif self.init_strategy == 'correlation':
            # Initialize proportional to correlation between feature and target
            correlations = []
            for j in range(n_features):
                # Simple correlation calculation
                corr = np.corrcoef(X[:, j], y)[0, 1]
                if np.isnan(corr):
                    corr = 0
                correlations.append(corr)
            self.w = np.array(correlations)
            # Normalize to avoid large initial values
            self.w = self.w / (np.linalg.norm(self.w) + 1e-9) * 0.01
        
        else:
            self.w = np.zeros(n_features)

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def decision_function(self, X):
        X_bias = self._add_bias(X)
        return np.dot(X_bias, self.w)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def fit(self, X, y):
        # Ensure y is {-1, 1}
        y = np.where(y <= 0, -1, 1)
        
        X_bias = self._add_bias(X)
        
        self._initialize_weights(X_bias, y)
        
        velocity = np.zeros_like(self.w)
        n_samples = X_bias.shape[0]
        
        # Recurrent quality estimation (exponential moving average of loss)
        ema_loss = None
        alpha_ema = 0.1
        
        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            
            if self.presentation_strategy == 'margin_abs':
                # Sort by absolute margin (uncertainty sampling)
                # M = y * <w, x>
                margins = y * np.dot(X_bias, self.w)
                abs_margins = np.abs(margins)
                sorted_indices = np.argsort(abs_margins)
                indices = sorted_indices
            elif self.presentation_strategy == 'random':
                np.random.shuffle(indices)
            
            epoch_loss = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                X_batch = X_bias[batch_idx]
                y_batch = y[batch_idx]
                
                # Nesterov lookahead
                if self.use_nesterov:
                    w_lookahead = self.w + self.momentum * velocity
                    y_pred = np.dot(X_batch, w_lookahead)
                    grad = self.loss_fn.gradient(y_batch, y_pred, X_batch)
                    if len(grad.shape) > 1:
                        grad = np.mean(grad, axis=0)
                else:
                    y_pred = np.dot(X_batch, self.w)
                    grad = self.loss_fn.gradient(y_batch, y_pred, X_batch)
                    if len(grad.shape) > 1:
                        grad = np.mean(grad, axis=0)
                
                # Add L2 regularization gradient: 2 * alpha * w
                grad += 2 * self.reg_alpha * self.w
                
                # Update velocity and weights
                velocity = self.momentum * velocity - self.learning_rate * grad
                self.w += velocity
                
                # Loss calculation for monitoring
                # Calculate on current batch or full? Batch is faster.
                current_loss = np.mean(self.loss_fn.loss(y_batch, y_pred)) + self.reg_alpha * np.sum(self.w**2)
                epoch_loss += current_loss * len(batch_idx)
                
                # Update EMA loss
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = (1 - alpha_ema) * ema_loss + alpha_ema * current_loss

            self.history['loss'].append(epoch_loss / n_samples)
            
        return self

    def get_margins(self, X, y):
        X_bias = self._add_bias(X)
        # Margin M = y * <w, x>
        return y * np.dot(X_bias, self.w)
