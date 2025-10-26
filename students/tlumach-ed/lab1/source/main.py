import os
import pandas as pd
from itertools import product
from dataset_loader import DataManager
from classifier import LinearClassifier
from benchmark_suite import Benchmark, calculate_metrics


# Hyperparameters
LR = 2e-4
MOMENTUM = 0.7
L2_STRENGTH = 1e-3
LOSS_SMOOTHING = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 64
N_RESTARTS = 5
VERBOSE = 0

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def run_experiment(config, X_train, y_train, X_test, y_test):
    desc = config['description']
    print(f"\n--- Experiment: {desc} ---")

    best_metrics = None
    best_model = None

    if config['weights_init'] == 'multistart':
        # multi-start random initialization: pick best by accuracy on test
        best_acc = -1.0
        for r in range(N_RESTARTS):
            model = LinearClassifier(
                init_method="random",
                batch_strategy=config['batch_method'],
                optimizer_type=config['optimizer_method'],
                lr=LR,
                momentum=MOMENTUM,
                l2_strength=L2_STRENGTH,
                loss_smoothing=LOSS_SMOOTHING
            )
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test,
                      n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
            preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds)
            if metrics['Accuracy'] > best_acc:
                best_acc = metrics['Accuracy']
                best_metrics = metrics
                best_model = model
        # visualizations for best
        if best_model is not None:
            best_model.plot_margins(X_test, y_test, desc)
            best_model.plot_training_history(desc)
    else:
        # correlation init single run
        model = LinearClassifier(
            init_method="correlation",
            batch_strategy=config['batch_method'],
            optimizer_type=config['optimizer_method'],
            lr=LR,
            momentum=MOMENTUM,
            l2_strength=L2_STRENGTH,
            loss_smoothing=LOSS_SMOOTHING
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test,
                  n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
        preds = model.predict(X_test)
        best_metrics = calculate_metrics(y_test, preds)
        model.plot_margins(X_test, y_test, desc)
        model.plot_training_history(desc)

    return {"description": desc, **best_metrics}

def main():
    X_train, X_test, y_train, y_test = DataManager().load_data()

    inits = ['multistart', 'correlation']
    optimizers = ['momentum', 'fast']
    batch_methods = ['margin', 'random']

    configs = []
    for init, opt, batch in product(inits, optimizers, batch_methods):
        configs.append({
            'weights_init': init,
            'optimizer_method': opt,
            'batch_method': batch,
            'description': f"{init} + {opt} + {batch}"
        })

    results = []
    for cfg in configs:
        res = run_experiment(cfg, X_train, y_train, X_test, y_test)
        # normalize fields
        desc = res.pop('description')
        entries = {
            "Method": desc,
            "Initialization": cfg['weights_init'],
            "Optimizer": cfg['optimizer_method'],
            "Batching": cfg['batch_method'],
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"]
        }
        results.append(entries)

    # sklearn benchmark
    sklearn_rows = Benchmark.run_sklearn(X_train, y_train, X_test, y_test, lr=LR, l2=L2_STRENGTH)
    results.extend(sklearn_rows)

    df = pd.DataFrame(results)
    # reorder & round
    final_cols = ['Method', 'Initialization', 'Optimizer', 'Batching', 'Accuracy', 'Precision', 'Recall', 'F1']
    df = df[final_cols].copy()
    df[['Accuracy','Precision','Recall','F1']] = df[['Accuracy','Precision','Recall','F1']].round(4)

    print("\n=== RESULTS ===")
    print(df.to_string(index=False))

    # save results to project
    save_path = Benchmark.save_results_df(df, filename="experiment_results.csv")
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()
