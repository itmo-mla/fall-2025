import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from svm import SVM
from metrics import Metrics

def generate_linearly_separable_data(n_samples=200, random_state=42):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        n_features=2,
        random_state=random_state,
        cluster_std=0.7
    )
    y = np.where(y == 0, -1, 1)
    return X, y

def plot_svm_linear(X, y, w, b, support_vectors, title="SVM с линейным ядром"):
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {-1: 'blue', 1: 'red'}
    for label in [-1, 1]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=colors[label], label=f'Класс {label}',
                  alpha=0.6, s=50)

    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
              s=200, facecolors='none', edgecolors='black',
              linewidths=2, label='Опорные вектора')

    xlim = ax.get_xlim()
    x_vals = np.linspace(xlim[0], xlim[1], 100)

    y_vals = (-b - w[0] * x_vals) / w[1]
    ax.plot(x_vals, y_vals, 'g-', linewidth=2, label='Разделяющая прямая')

    y_vals_plus = (-b + 1 - w[0] * x_vals) / w[1]
    ax.plot(x_vals, y_vals_plus, 'g--', linewidth=1.5, alpha=0.7)

    y_vals_minus = (-b - 1 - w[0] * x_vals) / w[1]
    ax.plot(x_vals, y_vals_minus, 'g--', linewidth=1.5, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    return fig, ax


def svm_linear_pipeline(X, y, model_type='custom'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

    match model_type:
        case 'sklearn':
            model = SVC(kernel='linear', C=1.0)
            model.fit(X_train, y_train)
            w = model.coef_[0]
            b = model.intercept_[0]
            support_vectors = model.support_vectors_
            title = 'Sklearn SVM'
        case 'custom':
            model = SVM(C=1.0, kernel='linear')
            model.fit(X_train, y_train)
            w = model.coef_[0]
            b = -model.b
            support_vectors = model.support_vectors
            title = 'CUSTOM SVM'
        case _:
            raise ValueError("Неизвестный тип модели")

    y_pred = model.predict(X_test)

    Metrics.print_all(model_type, y_test, y_pred)

    fig, ax = plot_svm_linear(X, y, w, b, support_vectors, title)

if __name__ == "__main__":
    X, y = generate_linearly_separable_data()
    svm_linear_pipeline(X, y, model_type='sklearn')
    svm_linear_pipeline(X, y, model_type='custom')
    plt.show()