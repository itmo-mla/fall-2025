import numpy as np

class BatchGenerator:
    def __init__(self, X, y, batch_size=1, shuffle=True, random_state=None,
                 sampling_strategy="uniform", margins=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.margins = margins
        self.num_samples = len(X)

        if random_state is not None:
            np.random.seed(random_state)

    def __iter__(self):
        X_data, y_data = self.X.copy(), self.y.copy()

        if self.sampling_strategy == "uniform" or self.margins is None:
            if self.shuffle:
                perm = np.random.permutation(self.num_samples)
                X_data, y_data = X_data[perm], y_data[perm]
        elif self.sampling_strategy in ["hard"]:
            abs_margins = np.abs(self.margins)
            
            if self.sampling_strategy == "hard":
                indices = np.argsort(abs_margins)
            
            X_data, y_data = X_data[indices], y_data[indices]

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            yield X_data[start_idx:end_idx], y_data[start_idx:end_idx]
