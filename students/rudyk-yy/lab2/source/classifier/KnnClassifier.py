import numpy as np

class KnnClassifier:
    def __init__(self, k = 3, ord = 2, kernel = "gaussian", weights = "uniform"):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.ord = ord
        self.kernel = kernel
        self.weights = weights
        self.best_accuracy=-1000
        self.accuracies = []

    def _metric(self, a,b):
        return np.linalg.norm(a - b, ord=self.ord)
    
    def _kernel(self, distance, h):
        if self.kernel == "gaussian":
            return np.exp(-2 * (distance / h) ** 2)

        elif self.kernel == "epanechnikov":
            u = distance / h
            return 0.75 * (1 - u ** 2) if abs(u) <= 1 else 0.0

        else:
            raise ValueError(f"Неподдерживаемый тип ядра: {self.kernel}")

    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n = X.shape[0]
        distances = np.zeros((X.shape[0], n))
        for i in range(X.shape[0]):
            for j in range(n):
                distances[i][j] = self._metric(X[i], self.X_train[j])
        self.distances = distances

    
        sorted_indices = [np.argsort(distances[i]) for i in range(n)]
        
        for k in range(2, min(n, 10)):
            y_pred = []
            for i in range(n):
                
                neighbor_idx = sorted_indices[i][1:k+1] 
                h = distances[i][neighbor_idx[-1]]
            
                
                weights = self._weights(distances[i], h, neighbor_idx)

                votes = np.zeros(len(np.unique(y)))
                
                for idx, cls in enumerate(np.unique(y)):
                    votes[idx] = weights[y[neighbor_idx] == cls].sum()
                y_pred.append(np.unique(y)[np.argmax(votes)])
            
            y_pred = np.array(y_pred)
            accuracy = np.sum(y_pred == y) / n
            print(accuracy)
            self.accuracies.append(accuracy)
            if k == 1 or accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.k = k
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(2, min(n, 10)), self.accuracies, marker='o')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs k (LOO)')
        plt.grid(True)
        plt.show()

    def _weights(self, distances, k_distance, neighbor_idx):
        match self.weights:
            case "uniform":
                return np.ones(len(neighbor_idx))
            case "kernel":
                return np.array([self._kernel(distances[j], k_distance) for j in neighbor_idx])


            
        
        
    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise Exception("Модель ещё не обучена.")
        
        y_pred = []
        classes = np.unique(self.y_train)
        
        for x in X:
            distances = np.array([self._metric(x, xi) for xi in self.X_train])
            neighbor_idx = np.argsort(distances)[:self.k] 
            h = distances[neighbor_idx[-1]]              
            
            weights = self._weights(distances, h, neighbor_idx)
            votes = np.zeros(len(classes))
            
            for idx, cls in enumerate(classes):
                votes[idx] = weights[self.y_train[neighbor_idx] == cls].sum()
            
            y_pred.append(classes[np.argmax(votes)])
        
        return np.array(y_pred)



    



    
