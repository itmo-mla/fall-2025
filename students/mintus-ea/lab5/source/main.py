import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import os

from core.logistic import LogisticRegression

def plot_loss(history, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    results_dir = "source/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data (Breast Cancer - Binary Classification)
    print("Loading Breast Cancer Dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Custom Implementation
    print("Training Custom Logistic Regression (Newton/IRLS)...")
    
    # We use small regularization for stability
    lr_custom = LogisticRegression(max_iter=50, lambda_reg=1.0)
    lr_custom.fit(X_train, y_train)
    
    plot_loss(lr_custom.loss_history, "Training Loss (Newton Method)", 
              os.path.join(results_dir, "loss_history.png"))
    
    y_pred_custom = lr_custom.predict(X_test)
    y_prob_custom = lr_custom.predict_proba(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    loss_custom = log_loss(y_test, y_prob_custom)
    
    print(f"Custom Model Accuracy: {acc_custom:.4f}")
    print(f"Custom Model Log Loss: {loss_custom:.4f}")
    print(f"Converged in {len(lr_custom.loss_history)} iterations")
    
    # 3. Sklearn Implementation
    print("Training Sklearn Logistic Regression...")
    
    C_val = 1.0 / lr_custom.lambda_reg
    sk_lr = SklearnLR(solver='newton-cg', C=C_val, fit_intercept=True, max_iter=100)
    sk_lr.fit(X_train, y_train)
    
    y_pred_sk = sk_lr.predict(X_test)
    y_prob_sk = sk_lr.predict_proba(X_test)[:, 1]
    acc_sk = accuracy_score(y_test, y_pred_sk)
    loss_sk = log_loss(y_test, y_prob_sk)
    
    print(f"Sklearn Model Accuracy: {acc_sk:.4f}")
    print(f"Sklearn Model Log Loss: {loss_sk:.4f}")
    
    # 4. Comparison
    w_custom = lr_custom.w[1:] # Skip intercept
    b_custom = lr_custom.w[0]
    
    w_sk = sk_lr.coef_.flatten()
    b_sk = sk_lr.intercept_[0]
    
    # Weights similarity
    cos_sim = np.dot(w_custom, w_sk) / (np.linalg.norm(w_custom) * np.linalg.norm(w_sk))
    print(f"Weights Cosine Similarity: {cos_sim:.6f}")
    print(f"Intercept Difference: {abs(b_custom - b_sk):.6f}")
    
    results = pd.DataFrame([{
        'Model': 'Custom (Newton)',
        'Accuracy': acc_custom,
        'Log Loss': loss_custom,
        'Iterations': len(lr_custom.loss_history)
    }, {
        'Model': 'Sklearn (Newton-CG)',
        'Accuracy': acc_sk,
        'Log Loss': loss_sk,
        'Iterations': sk_lr.n_iter_[0]
    }])
    
    results.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    print("\nResults:")
    print(results)

if __name__ == "__main__":
    main()
