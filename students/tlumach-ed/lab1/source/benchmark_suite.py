import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def calculate_metrics(y_true, y_pred):
    # y_true/y_pred can be in {-1,1}; sklearn metrics accept that
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1))
    }

class Benchmark:
    @staticmethod
    def run_sklearn(X_train, y_train, X_test, y_test, lr=1e-4, l2=1e-3):
        methods = {
            "Sklearn Logistic (SGD)": SGDClassifier(loss="log_loss", penalty="l2", alpha=l2,
                                                   eta0=lr, learning_rate="constant", max_iter=100, random_state=42),
            "Sklearn Linear SVM (SGD)": SGDClassifier(loss="hinge", penalty="l2", alpha=l2,
                                                      eta0=lr, learning_rate="constant", max_iter=100, random_state=42)
        }
        results = []
        for name, model in methods.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            m = calculate_metrics(y_test, preds)
            results.append({"Method": name, "Initialization": "sklearn", "Optimizer": "sklearn", "Batching": "sklearn", **m})
        return results

    @staticmethod
    def save_results_df(df: pd.DataFrame, filename: str = "experiment_results.csv"):
        path = os.path.join(os.path.dirname(__file__), filename)
        df.to_csv(path, index=False)
        return path
