import numpy as np
from typing import Callable, List
import math
from tqdm import tqdm

MIN_LEFT_SAMPLES = 20

class SimpleKNNClassifier:
    def __init__(self, k=None, ord=None, ker='gaussian'):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.shrinked_X_train = None
        self.shrinked_y_train = None
        self.distance_matrix = None 
        self.ord = ord
        self._sorted_indices = None
        self._sorted_y = None
        self.ref_elements_indices = None
        self.ref_elements_indices_history = []
        self.ker = ker

    def _distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=self.ord)
    
    def _ker(self, dist):
        if self.ker == 'gaussian':
            return np.exp(-2 * dist**2)
        elif self.ker == 'rectangular':
            return int(abs(dist) <= 1)
        elif self.ker == 'triangular':
            return 1 - abs(dist)
        elif self.ker == 'epanechnikov':
            return 1 - dist**2
        else:
            raise ValueError('Incorrect kernel')
    
    def _access_k_result(self, k):
        """ Access the result of the k-nearest neighbors algorithm with LOO"""

        correct_predictions = 0
        for i in range(len(self.X_train)):
            classes = {}
            dist_to_k = self.distance_matrix[i][self._sorted_indices[i][k]]
            for j in range(1, len(self.X_train)):
                weight = self._ker(self.distance_matrix[i][self._sorted_indices[i][j]] / dist_to_k)
                classes[self._sorted_y[i][j]] = classes.get(self._sorted_y[i][j], 0) + weight
            correct_predictions += max(classes, key=classes.get) == self.y_train[i]
        return correct_predictions / len(self.X_train)

    def fit(self, X_train, y_train, k_selection_callback: Callable[[List[float]], None] = None):
        """ 
            Simple KNN classifier with k-nearest neighbors algorithm
            k_selection_callback: Callable[[List[float]], None] = None - callback to select optimal k
            X_train: np.ndarray - training data
            y_train: np.ndarray - training labels
            k: int - number of neighbors
            ord: int - order of the norm
        """

        # init data
        self.X_train = []
        self.y_train = []
        for x, y in zip(X_train, y_train):
            self.X_train.append(x)
            self.y_train.append(y)
        self.X_train = np.array(self.X_train)
        self.ref_elements_indices = np.arange(len(self.X_train))
        self.y_train = np.array(self.y_train)
        self.distance_matrix = np.zeros((len(self.X_train), len(self.X_train)))

        # init distance matrix with sorting
        for i in range(len(self.X_train)):
            for j in range(len(self.X_train)):
                self.distance_matrix[i, j] = self._distance(self.X_train[i], self.X_train[j])
        
        self._sorted_indices = np.zeros((len(self.X_train), len(self.X_train)), dtype=int)
        self._sorted_y = np.zeros((len(self.X_train), len(self.X_train)))
        
        for i in range(len(self.X_train)):
            self._sorted_indices[i] = np.argsort(self.distance_matrix[i])
            sorted_labels = []
            for idx in self._sorted_indices[i]:
                sorted_labels.append(self.y_train[idx])
            self._sorted_y[i] = np.array(sorted_labels)

        # select optimal k
        if self.k is None:
            k_results = []
            for k in range(1, min(len(self.X_train), 50)):
                result = self._access_k_result(k)
                k_results.append(result)
            max_k_index = np.argmax(k_results)
            self.k = max_k_index + 1
            if k_selection_callback:
                k_selection_callback(k_results)

    def predict(self, X_test):
        """ 
            Predict the class of the data
            X_test: np.ndarray - data to predict
        """
    
        y_pred = []
        used_X_train = self.X_train if self.shrinked_X_train is None else self.shrinked_X_train
        used_y_train = self.y_train if self.shrinked_y_train is None else self.shrinked_y_train
        for x_test in X_test:
            distances = []
            for i in range(len(used_X_train)):
                dist = self._distance(x_test, used_X_train[i])
                distances.append((dist, used_y_train[i]))
            distances.sort(key=lambda x: x[0])
            classes = {}
            for dist, label in distances:
                weight = self._ker(dist / distances[self.k][0])
                classes[label] = classes.get(label, 0) + weight
            y_pred.append(max(classes, key=classes.get))
        return np.array(y_pred)

    def _comb_factor(self, L, l, m):
        return math.comb(L - 1 - m, l - 1) / math.comb(L - 1, l - 1)

    def _ccv_on_ref_elements(self, _ref_elements_indices):
        """
            Complete cross-validation on reference elements
        """
        MAX_CONTROL_SIZE = 3
        m_size = min(MAX_CONTROL_SIZE, len(_ref_elements_indices))
        p = np.zeros(m_size)
        
        ref_elements_array = np.array(_ref_elements_indices)
        ref_mask = np.zeros(len(self.X_train))
        ref_mask[ref_elements_array] = True
        
        for i in range(len(self.X_train)):
            sorted_idx = self._sorted_indices[i]
            is_ref_element = ref_mask[sorted_idx]
            ref_positions = np.where(is_ref_element)[0]
            for m in range(1, m_size):
                if m <= len(ref_positions):
                    mth_ref_pos = ref_positions[m - 1]
                    if mth_ref_pos == 0:
                        mth_ref_pos = ref_positions[m]
                    mth_ref_idx = sorted_idx[mth_ref_pos]
                    p[m] += self.y_train[mth_ref_idx] != self.y_train[i]
        
        for m in range(1, m_size):
            p[m] = p[m] * self._comb_factor(len(self.X_train), len(self.X_train) - m_size, m)
        return np.sum(p)

    def adjust_ref_by_ccv(self, ccv_callback: Callable[[float], None]):
        """
            Iteratively remove elements from reference list by minimizing CCV score
        """
        self.ref_elements_indices_history = []
        MAX_ITERATIONS = len(self.X_train) - MIN_LEFT_SAMPLES
        for _ in tqdm(range(MAX_ITERATIONS)):
            ref_elements_indices = list(self.ref_elements_indices)
            best_ccv_score = float('inf')
            for i in range(len(ref_elements_indices)):
                new_ref_elements_indices = ref_elements_indices[:i] + ref_elements_indices[i+1:]
                ccv_score = self._ccv_on_ref_elements(new_ref_elements_indices)
                if ccv_score < best_ccv_score:
                    best_ccv_score = ccv_score
                    best_ref_elements_indices = new_ref_elements_indices
            self.ref_elements_indices = best_ref_elements_indices
            self.ref_elements_indices_history.append(best_ref_elements_indices)
            if ccv_callback:
                ccv_callback(best_ccv_score)

    def shrink_x_by_ref_elements(self, removed_samples):
        X_train_ref = []
        y_train_ref = []
        for i in self.ref_elements_indices_history[removed_samples]:
            X_train_ref.append(self.X_train[i])
            y_train_ref.append(self.y_train[i])
        self.shrinked_X_train = np.array(X_train_ref)
        self.shrinked_y_train = np.array(y_train_ref)
