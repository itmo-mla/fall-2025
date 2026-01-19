import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(svm, X, y, kernel_type="linear"):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, levels=[-1,0,1])
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.title(f"SVM Decision Boundary ({kernel_type})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
