import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from svm import SVM
from metrics import Metrics


def generate_circles_data(n_samples=300, random_state=42):
    """
    Генерирует концентрические круги - классический пример нелинейной разделимости.

    Идеально подходит для RBF (гаусовского) ядра!
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=0.01,  # Меньше шума для лучшей визуализации
        random_state=random_state
    )
    # Преобразуем метки из {0, 1} в {-1, 1}
    y = np.where(y == 0, -1, 1)
    return X, y


def plot_svm_rbf(X, y, decision_function, support_vectors, title="SVM с RBF ядром (Гаусовское)"):

    fig, ax = plt.subplots(figsize=(12, 10))

    xlim = (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
    ylim = (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),
                         np.linspace(ylim[0], ylim[1], 300))

    Z = decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
              colors=['orange', 'darkred', 'orange'],
              linestyles=['--', '-', '--'],
              linewidths=[2, 2.5, 2],
              alpha=0.8)

    ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
               colors=['lightblue', 'lightcoral'],
               alpha=0.25)

    colors = {-1: 'blue', 1: 'red'}
    labels = {-1: 'Внешний круг (класс -1)', 1: 'Внутренний круг (класс +1)'}

    for label in [-1, 1]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=colors[label], label=labels[label],
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
              s=300, facecolors='none', edgecolors='darkgreen',
              linewidths=2.5, label='Опорные вектора', zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    return fig, ax


def svm_rbf_pipeline(X, y, model_type='custom'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

    match model_type:
        case 'sklearn':
            model = SVC(kernel='rbf', C=1.0, gamma='auto')
            model.fit(X_train, y_train)
            support_vectors = model.support_vectors_
            title = 'Sklearn SVM'
        case 'custom':
            model = SVM(C=1.0, kernel='rbf', gamma=0.5)
            model.fit(X_train, y_train)
            support_vectors = model.support_vectors
            title = 'CUSTOM SVM'
        case _:
            raise ValueError("Неизвестный тип модели")

    y_pred = model.predict(X_test)
    
    Metrics.print_all(model_type, y_test, y_pred)
    print(f"Опорных векторов: {len(support_vectors)} ({100*len(support_vectors)/len(X):.1f}%)\n")


    fig, ax = plot_svm_rbf(X, y, model.decision_function, support_vectors, title)


if __name__ == "__main__":
    X, y = generate_circles_data(n_samples=300)
    svm_rbf_pipeline(X, y, model_type='sklearn')
    svm_rbf_pipeline(X, y, model_type='custom')
    plt.show()