import numpy as np

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(np.array(y_pred) == np.array(y_true))

    @staticmethod
    def precision(y_true, y_pred):
        y_true = np.array(y_true).astype(np.int32)
        y_pred = np.array(y_pred).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == -1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred):
        y_true = np.array(y_true).astype(np.int32)
        y_pred = np.array(y_pred).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == -1) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1(y_true, y_pred):
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        y_true = np.array(y_true).astype(np.int32)
        y_pred = np.array(y_pred).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == -1) & (y_true == -1))
        fp = np.sum((y_pred == 1) & (y_true == -1))
        fn = np.sum((y_pred == -1) & (y_true == 1))
        return np.array([[tn, fp],
                         [fn, tp]])

    @staticmethod
    def print_all(name, y_true, y_pred):
        print(f"\n------------{name}------------")
        print("Accuracy:", Metrics.accuracy(y_true, y_pred))
        print("Confusion Matrix:\n", Metrics.confusion_matrix(y_true, y_pred))
        print("Precision:", Metrics.precision(y_true, y_pred))
        print("Recall:", Metrics.recall(y_true, y_pred))
        print("F1-score:", Metrics.f1(y_true, y_pred))
        print(f"------------{name}------------\n")