import inspect
from batch.batch_generator import BatchGenerator
from module.evaluator import *
from module.optimizer import *
from module.loss import *
from module.margin import *

def create_from_registry(name, registry, **kwargs):
    if name not in registry:
        raise KeyError(f"Неизвестное имя: {name}. Доступные: {list(registry.keys())}")
    
    cls = registry[name]
    sig = inspect.signature(cls.__init__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid_args)

class BinaryLinearClassification:
    def __init__(self, 
                 loss_function="hinge", 
                 optimizer="sgd", 
                 evaluator="reccurent",
                 margin:BaseMargin=None,
                 epochs=100, 
                 batch_size=1,
                 sampling_strategy="hard",
                 weight_init="random",
                 random_state=42,
                 **kwargs):
        
        self.loss_function = create_from_registry(loss_function, LOSSES, margin=margin, **kwargs)
        self.optimizer = create_from_registry(optimizer, OPTIMIZERS, **kwargs)
        self.evaluator = create_from_registry(evaluator, EVALUATORS, **kwargs)
        self.margin = margin
        self.epochs = epochs
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.weight_init = weight_init
        self.random_state = random_state
        self.weights = None

    def forward(self, X, weights):
        return X.dot(weights)

    def _init_weights(self, X, y):
        X_bias = np.c_[-np.ones((len(X), 1)), X]
        if self.weight_init == "correlation":
            corrs = np.corrcoef(X_bias.T, y)[-1, :-1]
            self.weights = np.nan_to_num(corrs)
        else:
            self.weights = np.random.randn(X_bias.shape[1])

    def fit(self, X, y, n_starts=1):
        best_weights = None
        best_loss = float('inf')

        # Мультистарт
        for start in range(n_starts):
            np.random.seed(self.random_state + start)
            self._init_weights(X, y)

            X_bias = np.c_[-np.ones((len(X), 1)), X]

            for _ in range(self.epochs):
                epoch_loss = []

                batch_gen = BatchGenerator(
                    X_bias,
                    y,
                    batch_size=self.batch_size,
                    shuffle=(self.sampling_strategy == "uniform"),
                    random_state=self.random_state,
                    sampling_strategy=self.sampling_strategy,
                    margins=self.margin.calculate(self.forward(X_bias, self.weights), y)
                )

                for X_batch, y_batch in batch_gen:
                    X_batch = np.atleast_2d(X_batch)
                    y_batch = np.atleast_1d(y_batch)

                    def gradient_function(weights):
                        y_pred = self.forward(X_batch, weights)
                        return self.loss_function.calculate_derivative(y_pred, y_batch, weights, X_batch)

                    y_pred = self.forward(X_batch, self.weights)
                    loss = self.loss_function.calculate_loss(y_pred, y_batch, self.weights)
                    epoch_loss.append(loss)
                    self.weights = self.optimizer.step(self.weights, gradient_function)

                self.evaluator.eval(np.mean(epoch_loss))
                self.evaluator.save_evaluation()

            if n_starts > 1:
                final_loss = np.mean(self.evaluator.get_evaluation_history())
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_weights = self.weights.copy()

        if n_starts > 1:
            self.weights = best_weights

        return self

    def predict(self, X):
        X_bias = np.c_[-np.ones((len(X), 1)), X]
        return np.sign(X_bias.dot(self.weights))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(self.evaluator.get_evaluation_history())),
            self.evaluator.get_evaluation_history(),
            marker='.', linestyle='-', color='b'
        )
        plt.title('Training Loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Exponential Training Loss')
        plt.grid(True)
        plt.show()