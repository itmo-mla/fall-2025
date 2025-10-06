import pandas as pd
import numpy as np
import random
import visualization
import math
import sklearn
from sklearn.datasets import load_breast_cancer

def standardize(X):
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  sigma[sigma == 0] = 1e-8
  X_standardized = (X - mu) / sigma
  return X_standardized

def count_margin(X, w, b, y):
        targets = predict(X, w, b)
        margins = targets * y
        return margins

def predict(X, w, b):
        return X @ w + b
def count_error(pred, y):
        return pred - y

def count_gradient(X, w, y):
        pred = predict(X, w)
        errors = count_error(pred, y)
        return 2 / len(X) * X.T @ errors

def quality_functional(pred, y):
        print(pred)
        print(y)
        errs_summ = 0
        for i in range(len(y)):
                if (pred[i] > 0 and y[i] > 0) or (pred[i] < 0 and y[i] < 0):
                        pass
                else:
                        errs_summ += 1
        return errs_summ / len(y)

def init_weights_correlation(n_features):
        w = np.zeros(n_features)
        for i in range(n_features):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                w[i] = correlation
        max_abs_corr = np.max(np.abs(w))
        if max_abs_corr != 0:
                w = w / max_abs_corr
        b = np.mean(y)
        return w, b

def init_weights_random(n_features):
        w = []
        for _ in range(n_features):
                w.append(random.uniform(-1, 1))
        #print(w)
        b = random.uniform(-1, 1)
        w = np.array(w)
        return w, b

def sample(X, y, size):
    indices = np.arange(len(X))
    inds = np.random.choice(indices, size=size, replace=False)
    X_subset = X[inds]
    y_subset = y[inds]
    return X_subset, y_subset

def sample_margin_module(X, y, size, w, b):
        margins = count_margin(X, w, b, y)
        indices = np.argsort(margins)
        sorted_X = X[indices]
        sorted_y = y[indices]
        count = 0
        setik = set()
        while count < size:
                a = 10
                while a > 5:
                        #Вероятность, что a < 5 примерно 0.99, тут мы не застрянем
                        a = np.random.exponential(1)
                a = a / 5
                a = a * size
                a = round(a)
                if a not in setik:
                        setik.add(a)
                        count += 1
        indices_from_setik = list(setik)
        sampled_X = sorted_X[indices_from_setik]
        sampled_y = sorted_y[indices_from_setik]
        return sampled_X, sampled_y

def count_mse_function(pred, y):
        errors = pred - y
        squared_errors = errors ** 2
        mse = np.mean(squared_errors)
        return mse

def count_errors(y_pred, y):
        return y_pred - y

def SGD(X, y, X_test, y_test, h, l, l2, l3, iterations, size_of_suddenly_subset, sampling_type, init_type):
        strings_num, features_num = X.shape
        tryes = 1
        if init_type == 'random':
                w, b = init_weights_random(features_num)
                tryes = 20
        elif init_type == 'correlation':
                w, b = init_weights_correlation(features_num)

        if sampling_type == 'random':
                Q, qy = sample(X, y, size_of_suddenly_subset)
        elif sampling_type == 'margin':
                Q, qy = sample_margin_module(X, y, size_of_suddenly_subset, w, b)
        pred = predict(Q, w, b)
        loss = count_mse_function(pred, qy)

        v = 0
        v_b = 0

        best_w = None
        best_b = None
        best_loss = math.inf
        best_accuracy = 0
        accuracy = 0

        for k in range(tryes):
                if k > 0:
                        #Тут нужно условие обновления
                        if loss < best_loss:
                                best_loss = loss
                                best_w = w
                                best_b = b
                        if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_w = w
                                best_b = b
                        w, b = init_weights_random(features_num)

                for i in range(iterations):
                        train_object, train_y = sample(X, y, 1)
                        train_prediction = predict(train_object, w, b)
                        train_loss = count_mse_function(train_prediction, train_y)
                        errors = count_errors(train_prediction, train_y)

                        w_gradient = 2 * np.dot(train_object.T, errors)
                        w_gradient += 2 * l2 * w
                        v = l3 * v + (1 - l3) * w_gradient
                        w = w - h * v

                        b_gradient = 2 * np.sum(errors)
                        b_gradient += 2 * l2 * b
                        v_b = l3 * v_b + (1 - l3) * b_gradient
                        b = b - h * v_b

                        pred = predict(Q, w, b)
                        new_loss = count_mse_function(pred, qy)
                        loss = l * new_loss + (1 - l) * loss

                        accuracy = qualitty_control(X_test, w, b, y_test, False)

                if tryes == 1:
                        best_w = w
                        best_b = b

                #print(train_loss)
        return best_w, best_b

def count_metrics(pred, real, viz):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(pred)):
                if real[i] < 0 and pred[i] < 0:
                        fp += 1
                if real[i] > 0 and pred[i] > 0:
                        tp += 1
                if real[i] > 0 and pred[i] < 0:
                        fn += 1
                if real[i] < 0 and pred[i] > 0:
                        fp += 1
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall =  tp / (tp + fn)
        f1_score = 2 * recall * precision / (recall + precision)
        if viz == True:
                print("Метрики подхода с линейной регессией")
                print("accuracy:  ", accuracy)
                print("precision: ", precision)
                print("recall:    ", recall)
                print("f1_score:  ", f1_score)
        return accuracy

def qualitty_control(X, w, b, y, viz):
        pred = predict(X, w, b)
        accuracy = count_metrics(pred, y, viz)

        margins = count_margin(X, w, b, y)
        if viz == True:
                visualization.visualize_margins_simple(np.sort(margins))

        return accuracy

breast_cancer = load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print(df.columns)
X = df.to_numpy()
X = standardize(X)
y = breast_cancer.target

print(y.shape)
print(sum(y))

y = np.where(y == 0, -1, y) #заменяем 0 на -1

#Делим выборку на тениовочные и тестовые данные
train_ratio = 0.8
num_samples = X.shape[0]
num_train = int(train_ratio * num_samples)
train_indices = np.random.choice(np.arange(num_samples), size=num_train, replace=False)
mask = np.isin(np.arange(num_samples), train_indices)
test_indices = np.arange(num_samples)[~mask]
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

#h - темп обучения l2 - регуляризация # l - для функционала качества (темп забывания)  # l3 - коэффициент в моментум
w, b = SGD(X_train, y_train, X_test, y_test, 0.001, 0.9, 0.1, 0.9, 1000, 10, 'random', 'random')
print("По самореализованной линейной регрессии")
qualitty_control(X_test, w, b, y_test, True)




from sklearn.linear_model import SGDRegressor

learning_rate = 0.001
momentum = 0.9
l1_ratio = 0.1
dampening = 0.9
batch_size = 10

model = SGDRegressor(
    loss='squared_error',
    penalty='elasticnet',
    alpha=l1_ratio,
    l1_ratio=0.9,
    learning_rate='constant',
    eta0=learning_rate,
    power_t=0.1,
    early_stopping=False,
    epsilon = dampening,
    max_iter=1000,
    random_state=None,
    warm_start=False,
    average=False
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("По модели sklearn")
count_metrics(y_pred, y_test, True)

















#Попытка сделать то же самое с лог регрессией

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def predict_logistic(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def count_logistic_loss(y_hat, y):
    m = len(y)
    cost = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
    return cost

def count_metrics_logistic(pred, real, viz):
        pred = (pred * 2) - 1
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(pred)):
                if real[i] < 0 and pred[i] < 0:
                        fp += 1
                if real[i] > 0 and pred[i] > 0:
                        tp += 1
                if real[i] > 0 and pred[i] < 0:
                        fn += 1
                if real[i] < 0 and pred[i] > 0:
                        fp += 1
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall =  tp / (tp + fn)
        f1_score = 2 * recall * precision / (recall + precision)
        if viz == True:
                print("accuracy:  ", accuracy)
                print("precision: ", precision)
                print("recall:    ", recall)
                print("f1_score:  ", f1_score)
        return accuracy

def count_margin_logistic(X, w, b, y):
        targets = predict_logistic(X, w, b)
        targets = (targets * 2) - 1
        y = np.where(y == 0, -1, y)
        margins = targets * y
        return margins
def qualitty_control_logistic(X, w, b, y, viz):
        pred = predict_logistic(X, w, b)
        accuracy = count_metrics_logistic(pred, y, viz)

        margins = count_margin_logistic(X, w, b, y)
        if viz == True:
                visualization.visualize_margins_simple(np.sort(margins))

        return accuracy


def SGD_logistic(X, y, X_test, y_test, h, l, l2, l3, iterations, size_of_suddenly_subset, sampling_type, init_type):
        strings_num, features_num = X.shape
        tryes = 1
        if init_type == 'random':
                w, b = init_weights_random(features_num)
                tryes = 20
        elif init_type == 'correlation':
                w, b = init_weights_correlation(features_num)

        if sampling_type == 'random':
                Q, qy = sample(X, y, size_of_suddenly_subset)
        elif sampling_type == 'margin':
                Q, qy = sample_margin_module(X, y, size_of_suddenly_subset, w, b)
        pred = predict_logistic(Q, w, b)

        loss = count_logistic_loss(pred, qy)

        v = 0
        v_b = 0

        best_w = None
        best_b = None
        best_loss = math.inf
        best_accuracy = 0
        accuracy = 0

        for k in range(tryes):
                if k > 0:
                        #Тут нужно условие обновления
                        if loss < best_loss:
                                best_loss = loss
                                best_w = w
                                best_b = b
                        if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_w = w
                                best_b = b
                        w, b = init_weights_random(features_num)

                for i in range(iterations):
                        train_object, train_y = sample(X, y, 1)
                        train_prediction = predict_logistic(train_object, w, b)
                        train_loss = count_logistic_loss(train_prediction, train_y)
                        errors = count_errors(train_prediction, train_y)

                        w_gradient = np.dot(train_object.T, errors) # Removed the 2* factor, no need in Logistic Regression
                        w_gradient += 2 * l2 * w
                        v = l3 * v + (1 - l3) * w_gradient
                        w = w - h * v

                        b_gradient = np.sum(errors) # Removed the 2* factor, no need in Logistic Regression
                        b_gradient += 2 * l2 * b
                        v_b = l3 * v_b + (1 - l3) * b_gradient
                        b = b - h * v_b

                        pred = predict_logistic(Q, w, b)
                        new_loss = count_logistic_loss(pred, qy)
                        loss = l * new_loss + (1 - l) * loss

                        accuracy = qualitty_control_logistic(X_test, w, b, y_test, False)

                if tryes == 1:
                        best_w = w
                        best_b = b

                #print(train_loss)
        return best_w, best_b

y_test = np.where(y_test == -1, 0, y_test)
y_train = np.where(y_train == -1, 0, y_train)

print("По самореализованной логистической регрессии")
w, b = SGD_logistic(X_train, y_train, X_test, y_test, 0.001, 0.9, 0.1, 0.9, 1000, 10, 'margin', 'correlation')
qualitty_control_logistic(X_test, w, b, y_test, True)

from sklearn.linear_model import SGDClassifier
learning_rate = 0.001
alpha = 0.1
l1_ratio = 0.9
max_iter = 1000

sgd_logreg = SGDClassifier(loss='log_loss',
                          penalty='elasticnet',
                          alpha=alpha,
                          l1_ratio = l1_ratio,
                          learning_rate='constant',
                          eta0=learning_rate,
                          max_iter=max_iter,
                          random_state=42,
                          shuffle=True)

sgd_logreg.fit(X_train, y_train)
y_pred = sgd_logreg.predict(X_test)
print("Метрики log регрессии из sklearn")
count_metrics_logistic(y_pred, y_test, True)
