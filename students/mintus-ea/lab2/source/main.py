import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

from core.knn import KNN
from core.kernels import gaussian_kernel
from core.evaluation import leave_one_out_error
from core.selection import STOLP

def plot_loo_errors(errors, title, save_path):
    k_values = list(errors.keys())
    err_values = list(errors.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, err_values, marker='o')
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('LOO Error')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_prototypes(X, y, selected_indices, title, save_path):
    # Project to 2D using PCA if dims > 2
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_proj = pca.fit_transform(X)
    else:
        X_proj = X
        
    plt.figure(figsize=(12, 8))
    
    # Plot all points faintly
    unique_labels = np.unique(y)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = (y == label)
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1], 
                    c=[colors[i]], alpha=0.3, label=f'Class {label} (All)', s=30)
        
    # Plot prototypes boldly
    X_proto = X_proj[selected_indices]
    y_proto = y[selected_indices]
    
    for i, label in enumerate(unique_labels):
        mask = (y_proto == label)
        if np.any(mask):
            plt.scatter(X_proto[mask, 0], X_proto[mask, 1], 
                        c=[colors[i]], marker='x', s=100, linewidth=2, label=f'Class {label} (Proto)')
            
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    results_dir = "source/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading Wine Dataset...")
    data = load_wine()
    X = data.data
    y = data.target
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split for final evaluation (though LOO is used for tuning on Train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. LOO for k selection
    print("Running LOO for k selection...")
    k_range = range(1, 21)
    loo_errors = leave_one_out_error(X_train, y_train, k_range, kernel=gaussian_kernel)
    
    best_k = min(loo_errors, key=loo_errors.get)
    print(f"Best k found: {best_k} (Error: {loo_errors[best_k]:.4f})")
    
    plot_loo_errors(loo_errors, "LOO Error vs k (Gaussian Kernel)", os.path.join(results_dir, "loo_errors.png"))
    
    # 3. Train Best Model and Evaluate
    print("Evaluating Best Model...")
    my_knn = KNN(k=best_k, kernel=gaussian_kernel, method='variable_window')
    my_knn.fit(X_train, y_train)
    y_pred_my = my_knn.predict(X_test)
    acc_my = accuracy_score(y_test, y_pred_my)
    print(f"My KNN Accuracy: {acc_my:.4f}")
    
    # 4. Compare with Sklearn
    print("Evaluating Sklearn KNN...")
    # Sklearn doesn't support variable parzen window easily out of box, 
    # but we compare with standard weighted KNN or just KNN
    sk_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance') # closest equivalent
    sk_knn.fit(X_train, y_train)
    y_pred_sk = sk_knn.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"Sklearn KNN (weights='distance') Accuracy: {acc_sk:.4f}")
    
    # 5. Prototype Selection (STOLP)
    print("Running STOLP Prototype Selection...")
    stolp = STOLP(k=best_k, kernel=gaussian_kernel)
    stolp.fit(X_train, y_train)
    
    selected_indices = stolp.get_prototypes()
    print(f"Selected {len(selected_indices)} prototypes out of {len(X_train)} training samples.")
    
    # Evaluate reduced model
    X_proto = X_train[selected_indices]
    y_proto = y_train[selected_indices]
    
    knn_proto = KNN(k=best_k, kernel=gaussian_kernel, method='variable_window')
    # Be careful: if prototypes are few, k might be too large
    safe_k = min(best_k, len(X_proto))
    knn_proto.k = safe_k
    
    knn_proto.fit(X_proto, y_proto)
    y_pred_proto = knn_proto.predict(X_test)
    acc_proto = accuracy_score(y_test, y_pred_proto)
    print(f"STOLP KNN Accuracy: {acc_proto:.4f} (with k={safe_k})")
    
    # Visualize
    plot_prototypes(X_train, y_train, selected_indices, 
                    f"STOLP Selection (Reduced to {len(selected_indices)}/{len(X_train)})", 
                    os.path.join(results_dir, "stolp_visualization.png"))
    
    # Save Metrics
    results = {
        'My_KNN_Acc': acc_my,
        'Sklearn_KNN_Acc': acc_sk,
        'STOLP_KNN_Acc': acc_proto,
        'Best_k': best_k,
        'Prototypes_Count': len(selected_indices),
        'Original_Count': len(X_train)
    }
    df = pd.DataFrame([results])
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("\nResults:")
    print(df)

if __name__ == "__main__":
    main()
