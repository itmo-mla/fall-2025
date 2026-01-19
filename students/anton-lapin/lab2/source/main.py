import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def fit_standardize(X):
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  sigma[sigma == 0] = 1e-8
  X_standardized = (X - mu) / sigma
  return X_standardized, mu, sigma

def standartize(X, mu, sigma):
    return (X - mu) / sigma

def vizualize_emperik_factor(emperik_factor):
    k_values = np.arange(1, len(emperik_factor) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, emperik_factor, marker='o', linestyle='-', color='blue')
    plt.title('Эмпирический риск в зависимости от k')
    plt.xlabel('Значение k')
    plt.ylabel('Эмпирический риск')
    plt.grid(True)
    plt.show()

def count_metrics(pred, real, viz):
    correct = 0
    incorrect = 0
    for i in range(len(pred)):
        if real[i] == pred[i]:
            correct += 1
        if real[i] != pred[i]:
            incorrect += 1

    accuracy = correct / len(pred)
    if viz == True:
        print("accuracy:  ", accuracy)

    return accuracy

#Под obj1 понимается тестовое значение
#Под obj2 понимается тренировочное значение
def evklid_dist(obj1, obj2):
    return np.sqrt(np.sum((obj1 - obj2)*(obj1 - obj2)))

def gauss_kernel(dist):
    return np.exp(-2*dist*dist)

def predict_parzen(X_train, y_train, X_test, k):
    predictions = []
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            dist = evklid_dist(X_test[i], X_train[j])
            distances.append((dist, y_train[j]))

        distances.sort(key=lambda x: x[0])
        h = distances[k][0] if k < len(distances) else distances[-1][0]

        class_weights = {}
        for dist, label in distances:
            r = dist / h
            weight = gauss_kernel(r)
            class_weights[label] = class_weights.get(label, 0) + weight

        predictions.append(max(class_weights, key=class_weights.get))
    return predictions

def LOO(min_k, max_k, X, y):
    emperik_factor = []
    best_good = 0
    best_k = min_k
    for k in range(min_k, max_k + 1):
        count_good = 0
        for i in range(len(X)):
            X_test = X[i].reshape(1,-1)
            y_test = y[i]
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            prediction = predict_parzen(X_train, y_train, X_test, k)
            if prediction[0] == y_test:
                count_good += 1
        if count_good > best_good:
            best_good = count_good
            best_k = k
        emperik_factor.append( (len(X_train) - count_good)/len(X_train) )
    return best_k, emperik_factor

iris = datasets.load_iris()
X = iris.data
y = iris.target

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

X_train, mu, sigma = fit_standardize(X_train)
X_test = standartize(X_test, mu, sigma)

best_k, emperik_factor = LOO(1, 25, X_train, y_train)
print("Наиболее подходящее значение k", best_k)
vizualize_emperik_factor(emperik_factor)

pred = predict_parzen(X_train, y_train, X_test, best_k)
#print(pred)
#print(y_test)
print("Результат написанного нами Парзена")
count_metrics(pred, y_test, True)


#sklearn с теми же параметрами отобранными по LOO, обычная классификация knn
from sklearn.neighbors import KNeighborsClassifier
knn_sklearn = KNeighborsClassifier(n_neighbors=best_k)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
accuracy_sklearn = count_metrics(y_pred_sklearn, y_test, False)
print("Эталонная реализация обычного KNN из sklearn")
print(accuracy_sklearn)


import math
def predict_1NN(X_train, y_train, X_test, y_test):
    min_rasst = math.inf
    ind = 0
    for i in range(len(X_train)):
        rasst = evklid_dist(X_test, X_train[i])
        if rasst < min_rasst:
            ind = i
            min_rasst = rasst
    if y_test == y_train[ind]:
        return 1
    return 0

numbers = np.arange(0, len(X_train))
def etalons_work(X_train, numbers, y_train):
    set_of_deleted = set()
    objects_exist = len(X_train)
    while objects_exist > 30:
        print(objects_exist)
        #Шаг 1 - оценим эмпирический риск
        correct = 0
        for i in range(len(X_train)):
            correct += predict_1NN(np.delete(X_train, i, axis=0), np.delete(y_train, i), X_train[i], y_train[i])
        ccv = correct/len(X_train)

        #Шаг 2 - будем делать это для всех бех 1 элемента
        best_ccv = ccv
        best_ind = 0
        best_progress = -1000
        for i in range(len(X_train)):
            X_train1 = np.delete(X_train, i, axis=0)
            y_train1 = np.delete(y_train, i)
            correct1 = 0
            for j in range(len(X_train) - 1):
                correct1 += predict_1NN(np.delete(X_train1, j, axis=0), np.delete(y_train1, j), X_train1[j], y_train1[j])
            ccv1 = correct1/(len(X_train) - 1)
            if best_progress < ccv1 - ccv:
                best_progress = ccv1 - ccv
            if ccv1 > best_ccv:
                best_ccv = ccv1
                best_ind = i #Какой по счету элемент удаляем

        if best_progress < 0:
            break

        set_of_deleted.add(numbers[best_ind])
        X_train = np.delete(X_train, best_ind, axis=0)
        y_train = np.delete(y_train, best_ind)
        numbers = np.delete(numbers, best_ind)
        objects_exist -= 1

    return numbers

numbers_of_etalons = etalons_work(X_train, numbers, y_train)
X_train_full = X_train
y_train_full = y_train
X_train = X_train[numbers_of_etalons]
y_train = y_train[numbers_of_etalons]

best_k, emperik_factor = LOO(1, 25, X_train, y_train)
print("Наиболее подходящее значение k", best_k)
vizualize_emperik_factor(emperik_factor)

pred = predict_parzen(X_train, y_train, X_test, best_k)
print("Результат написанного нами Парзена c эталонами")
count_metrics(pred, y_test, True)



knn_sklearn = KNeighborsClassifier(n_neighbors=best_k)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
accuracy_sklearn = count_metrics(y_pred_sklearn, y_test, False)
print("Эталонная реализация обычного KNN из sklearn с эталонами")
print(accuracy_sklearn)


#Визуализация

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_full)

etalon_mask = np.zeros(len(X_train_full), dtype=bool)
etalon_mask[numbers_of_etalons] = True
deleted_mask = ~etalon_mask

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
           c=y_train_full, cmap='viridis', alpha=0.6, s=30)
plt.title('Исходная выборка')
plt.xlabel('PCA компонента 1')
plt.ylabel('PCA компонента 2')

plt.subplot(1, 2, 2)
plt.scatter(X_train_pca[deleted_mask, 0], X_train_pca[deleted_mask, 1],
           c='black', label='Удалённые')
plt.scatter(X_train_pca[etalon_mask, 0], X_train_pca[etalon_mask, 1],
           c=y_train_full[etalon_mask], cmap='viridis', edgecolors='red', linewidth=1.5, label='Эталоны')
plt.title('Эталоны и удалённые объекты')
plt.xlabel('PCA компонента 1')
plt.ylabel('PCA компонента 2')
plt.legend()

plt.tight_layout()
plt.show()
