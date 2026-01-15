from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, pos_label=1), 3),
        "Recall": round(recall_score(y_true, y_pred, pos_label=1), 3),
        "F1": round(f1_score(y_true, y_pred, pos_label=1), 3)
    }
