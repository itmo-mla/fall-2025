import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from core.model import LinearClassifier
from core.utils import calculate_metrics, plot_history, plot_margins

def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Convert 0/1 to -1/1
    y = np.where(y == 0, -1, 1)
    return X, y

def run_experiment(name, model_params, X_train, y_train, X_test, y_test, results_dir):
    print(f"Running experiment: {name}")
    
    if model_params.get('init_strategy') == 'multi_start':
        best_loss = float('inf')
        best_model = None
        n_starts = 5
        for i in range(n_starts):
            print(f"  Start {i+1}/{n_starts}")
            model = LinearClassifier(**{k: v for k, v in model_params.items() if k != 'init_strategy'}, init_strategy='random')
            model.fit(X_train, y_train)
            final_loss = model.history['loss'][-1]
            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model
        model = best_model
    else:
        model = LinearClassifier(**model_params)
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    print(f"  Train Accuracy: {metrics_train['accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics_test['accuracy']:.4f}")
    
    # Plots
    plot_history(model.history, title=f"{name} - Training Loss", 
                 save_path=os.path.join(results_dir, f"{name}_loss.png"))
    
    margins_train = model.get_margins(X_train, y_train)
    plot_margins(margins_train, title=f"{name} - Margin Distribution (Train)", 
                 save_path=os.path.join(results_dir, f"{name}_margins.png"))
    
    return metrics_test

def main():
    # Setup
    results_dir = "source/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Data
    X, y = load_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Basic SGD with Momentum, L2 (Target implementation)
    params_1 = {
        'learning_rate': 0.001,
        'n_epochs': 200,
        'batch_size': 10,
        'momentum': 0.9,
        'reg_alpha': 0.01,
        'use_nesterov': True,
        'init_strategy': 'random',
        'presentation_strategy': 'random'
    }
    results['SGD_Momentum_L2'] = run_experiment('SGD_Momentum_L2', params_1, X_train, y_train, X_test, y_test, results_dir)
    
    # 2. Steepest Gradient Descent (Full Batch)
    params_2 = params_1.copy()
    params_2['batch_size'] = len(X_train)
    results['Steepest_GD'] = run_experiment('Steepest_GD', params_2, X_train, y_train, X_test, y_test, results_dir)
    
    # 3. Initialization via Correlation
    params_3 = params_1.copy()
    params_3['init_strategy'] = 'correlation'
    results['Init_Correlation'] = run_experiment('Init_Correlation', params_3, X_train, y_train, X_test, y_test, results_dir)
    
    # 4. Multi-start Initialization
    params_4 = params_1.copy()
    params_4['init_strategy'] = 'multi_start'
    results['Init_MultiStart'] = run_experiment('Init_MultiStart', params_4, X_train, y_train, X_test, y_test, results_dir)
    
    # 5. Margin-based Presentation
    params_5 = params_1.copy()
    params_5['presentation_strategy'] = 'margin_abs'
    results['Presentation_Margin'] = run_experiment('Presentation_Margin', params_5, X_train, y_train, X_test, y_test, results_dir)
    
    # 6. Comparison with Sklearn
    print("Running Sklearn Baseline...")
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    metrics_svc = calculate_metrics(y_test, y_pred_svc)
    results['Sklearn_SVC'] = metrics_svc
    print(f"  SVC Test Accuracy: {metrics_svc['accuracy']:.4f}")
    
    # Save results to table
    df_res = pd.DataFrame(results).T
    print("\nFinal Comparison:")
    print(df_res)
    df_res.to_csv(os.path.join(results_dir, "comparison_metrics.csv"))

if __name__ == "__main__":
    main()
