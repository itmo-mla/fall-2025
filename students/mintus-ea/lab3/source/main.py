import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import os

from core.svm import SVM

def plot_decision_boundary(model, X, y, title, save_path):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#FFAAAA', '#AAAAFF'], alpha=0.2)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    
    # Plot support vectors if available in custom model
    if hasattr(model, 'support_vectors') and model.support_vectors is not None:
        plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                    s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    results_dir = "source/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Dataset Generation (Linearly Separable)
    print("Generating Linearly Separable Data...")
    X_lin, y_lin = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=1, n_clusters_per_class=1)
    y_lin = np.where(y_lin == 0, -1, 1)
    
    # 2. Linear SVM
    print("Training Linear SVM...")
    svm_lin = SVM(kernel='linear', C=1.0)
    svm_lin.fit(X_lin, y_lin)
    
    plot_decision_boundary(svm_lin, X_lin, y_lin, "Linear SVM (Custom)", 
                           os.path.join(results_dir, "linear_svm_custom.png"))
    
    # Compare with Sklearn
    print("Training Sklearn Linear SVM...")
    sk_lin = SVC(kernel='linear', C=1.0)
    sk_lin.fit(X_lin, y_lin)
    plot_decision_boundary(sk_lin, X_lin, y_lin, "Linear SVM (Sklearn)", 
                           os.path.join(results_dir, "linear_svm_sklearn.png"))

    # 3. Non-linear Dataset (Moons)
    print("Generating Non-linear Data (Moons)...")
    X_moon, y_moon = make_moons(n_samples=100, noise=0.1, random_state=42)
    y_moon = np.where(y_moon == 0, -1, 1)
    
    # 4. RBF SVM (Kernel Trick)
    print("Training RBF SVM...")
    svm_rbf = SVM(kernel='rbf', C=1.0, gamma=0.5)
    svm_rbf.fit(X_moon, y_moon)
    
    plot_decision_boundary(svm_rbf, X_moon, y_moon, "RBF SVM (Custom)", 
                           os.path.join(results_dir, "rbf_svm_custom.png"))
    
    # Compare with Sklearn
    print("Training Sklearn RBF SVM...")
    sk_rbf = SVC(kernel='rbf', C=1.0, gamma=0.5)
    sk_rbf.fit(X_moon, y_moon)
    plot_decision_boundary(sk_rbf, X_moon, y_moon, "RBF SVM (Sklearn)", 
                           os.path.join(results_dir, "rbf_svm_sklearn.png"))
    
    # 5. Metrics Comparison
    results = []
    
    # Linear
    y_pred_lin_custom = svm_lin.predict(X_lin)
    y_pred_lin_sk = sk_lin.predict(X_lin)
    results.append({
        'Dataset': 'Linear',
        'Model': 'Custom SVM',
        'Accuracy': accuracy_score(y_lin, y_pred_lin_custom),
        'Support Vectors': len(svm_lin.support_vectors)
    })
    results.append({
        'Dataset': 'Linear',
        'Model': 'Sklearn SVC',
        'Accuracy': accuracy_score(y_lin, y_pred_lin_sk),
        'Support Vectors': len(sk_lin.support_vectors_)
    })
    
    # RBF
    y_pred_rbf_custom = svm_rbf.predict(X_moon)
    y_pred_rbf_sk = sk_rbf.predict(X_moon)
    results.append({
        'Dataset': 'Moons (RBF)',
        'Model': 'Custom SVM',
        'Accuracy': accuracy_score(y_moon, y_pred_rbf_custom),
        'Support Vectors': len(svm_rbf.support_vectors)
    })
    results.append({
        'Dataset': 'Moons (RBF)',
        'Model': 'Sklearn SVC',
        'Accuracy': accuracy_score(y_moon, y_pred_rbf_sk),
        'Support Vectors': len(sk_rbf.support_vectors_)
    })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    print("\nResults:")
    print(df)

if __name__ == "__main__":
    main()
