import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import os

from core.pca import PCA

def plot_variance(pca_model, title, save_path):
    plt.figure(figsize=(10, 6))
    
    # Cumulative Variance
    cum_var = np.cumsum(pca_model.explained_variance_ratio_)
    
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
    plt.axhline(y=0.99, color='g', linestyle=':', label='99% Variance')
    
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_projection(X_proj, y, title, save_path):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Target')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    results_dir = "source/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Dataset (Digits - high dimensional, good for PCA visualization)
    print("Loading Digits Dataset...")
    data = load_digits()
    X = data.data
    y = data.target
    
    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    
    # 2. PCA Implementation
    print("Running Custom PCA...")
    pca = PCA() # Keep all components first
    pca.fit(X_std)
    
    # 3. Determine Effective Dimension
    # Scree plot / Cumulative Variance
    plot_variance(pca, "Cumulative Explained Variance (Digits)", 
                  os.path.join(results_dir, "variance_plot.png"))
    
    # Find components for 95% variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cum_var >= 0.95) + 1
    print(f"Number of components for 95% variance: {n_95}")
    
    # 4. Transform to 2D and Visualize
    pca_2d = PCA(n_components=2)
    X_proj = pca_2d.fit_transform(X_std)
    plot_projection(X_proj, y, "PCA 2D Projection (Custom)", 
                    os.path.join(results_dir, "pca_projection_custom.png"))
    
    # 5. Comparison with Sklearn
    print("Comparing with Sklearn...")
    sk_pca = SklearnPCA(n_components=2)
    X_proj_sk = sk_pca.fit_transform(X_std)
    
    plot_projection(X_proj_sk, y, "PCA 2D Projection (Sklearn)", 
                    os.path.join(results_dir, "pca_projection_sklearn.png"))
    

    # Re-run full PCA for comparison of components
    pca_full = PCA(n_components=None)
    pca_full.fit(X_std)
    
    sk_pca_full = SklearnPCA(n_components=None)
    sk_pca_full.fit(X_std)
    
    # Compare Explained Variance Ratio
    var_diff = np.max(np.abs(pca_full.explained_variance_ratio_ - sk_pca_full.explained_variance_ratio_))
    print(f"Max difference in explained variance ratio: {var_diff:.2e}")
    

    print("\nChecking components similarity (Cosine Similarity of first 5 components):")
    similarities = []
    for i in range(5):
        v1 = pca_full.components_[i]
        v2 = sk_pca_full.components_[i]
        sim = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        similarities.append(sim)
        print(f"  Component {i+1}: {sim:.6f}")
        
    df = pd.DataFrame({
        'Component': range(1, 6),
        'Cosine_Similarity': similarities
    })
    df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    
    if np.all(np.array(similarities) > 0.99):
        print("\nSUCCESS: Components match (ignoring sign flips).")
    else:
        print("\nWARNING: Components do not match perfectly.")

if __name__ == "__main__":
    main()
