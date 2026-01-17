import os
import numpy as np
from pathlib import Path
from config import Config
from data.datasets.drug_dataset import DrugDataset
from data.datasets.iris_dataset import IrisDataset
from data.datasets.wine_dataset import WineDataset
from data.datasets.breast_cancer_dataset import BreastCancerDataset
from model.knn import KNN
from algorithms.loo import LeaveOneOut
from algorithms.prototype_selection import PrototypeSelection
from utils.metrics import ClassificationMetrics
from utils.visualization import Visualizer
from utils.report import ReportGenerator
from sklearn.neighbors import KNeighborsClassifier


class LabWorkRunner:
    def __init__(self, config: Config):
        self.prototype_metrics = None
        self.sklearn_metrics = None
        self.our_metrics = None
        self.config = config
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        self.dataset = None
        self.loo_results = None
        self.best_k = None

        if config.save_plots: Path(config.plots_dir).mkdir(exist_ok=True)

    def load_dataset(self):
        print("Загрузка датасета...\n")

        dataset_name = self.config.dataset_name.lower()

        dataset_classes = {
            'iris': IrisDataset,
            'wine': WineDataset,
            'breast_cancer': BreastCancerDataset,
            'cancer': BreastCancerDataset,
            'drug': DrugDataset
        }

        if dataset_name not in dataset_classes:
            available = ', '.join(dataset_classes.keys())
            raise ValueError(
                f"Неизвестный датасет: {self.config.dataset_name}. "
                f"Доступные датасеты: {available}"
            )

        dataset_class = dataset_classes[dataset_name]
        self.dataset = dataset_class(seed=self.config.seed)

        self.dataset.split_and_scale(
            test_size=self.config.test_size
        )

        print(f"Размер обучающей выборки: {len(self.dataset.X_train)}")
        print(f"Размер тестовой выборки: {len(self.dataset.X_test)}")
        print(f"Количество признаков: {self.dataset.X_train.shape[1]}")
        print(f"Количество классов: {len(np.unique(self.dataset.y_train))}")
        print()

    def find_best_k_with_loo(self):
        print("Подбор параметра k методом скользящего контроля (LOO)...\n")

        loo = LeaveOneOut(self.config.loo_k_range)

        best_k, best_error, errors = loo.evaluate(
            self.dataset.X_train,
            self.dataset.y_train
        )

        self.loo_results = {
            'best_k': best_k,
            'best_error': best_error,
            'k_range': self.config.loo_k_range,
            'errors': errors
        }
        self.best_k = best_k

        print(f"Лучшее значение k: {best_k}")
        print(f"Эмпирический риск при k={best_k}: {best_error:.4f}")
        print()

        if self.config.save_plots:
            save_path = os.path.join(self.config.plots_dir, 'empirical_risk.png')
            self.visualizer.plot_empirical_risk(
                self.config.loo_k_range,
                errors,
                save_path=save_path
            )

    @staticmethod
    def train_knn(X_train, y_train, k: int) -> KNN:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        return knn

    @staticmethod
    def evaluate_knn(knn: KNN, X: np.ndarray, y: np.ndarray, name: str) -> dict:
        y_pred = knn.predict(X)
        metrics = ClassificationMetrics.get_all_metrics(y, y_pred)

        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print()

        return metrics

    def train_our(self) -> None:
        k_to_use = self.best_k if self.best_k is not None else (self.config.k_range[len(self.config.k_range) // 2] if self.config.k_range else 5)
        our_knn = self.train_knn(self.dataset.X_train, self.dataset.y_train, k_to_use)
        our_metrics = self.evaluate_knn(our_knn, self.dataset.X_test, self.dataset.y_test, "KNN")
        y_pred = our_knn.predict(self.dataset.X_test)
        self.visualizer.plot_confusion_matrix(
            self.dataset.y_test,
            y_pred,
            title='Матрица ошибок: KNN без отбора эталонов',
            save_path=os.path.join(self.config.plots_dir, 'confusion_matrix_full.png')
        )
        self.our_metrics = our_metrics

    def train_sklearn(self):
        k_to_use = self.best_k if self.best_k is not None else (self.config.k_range[len(self.config.k_range) // 2] if self.config.k_range else 5)
        sklearn_knn = KNeighborsClassifier(n_neighbors=k_to_use)
        sklearn_knn.fit(self.dataset.X_train, self.dataset.y_train)
        sklearn_pred = sklearn_knn.predict(self.dataset.X_test)
        sklearn_metrics = ClassificationMetrics.get_all_metrics(self.dataset.y_test, sklearn_pred)

        print("Sklearn KNeighborsClassifier:")
        print(f"  Accuracy: {sklearn_metrics['accuracy']:.4f}")
        print(f"  Precision: {sklearn_metrics['precision']:.4f}")
        print(f"  Recall: {sklearn_metrics['recall']:.4f}")
        print(f"  F1-score: {sklearn_metrics['f1']:.4f}")
        print()

        self.sklearn_metrics = sklearn_metrics

    def prototype_selection_experiment(self):
        print("Эксперимент с отбором эталонов...")

        ps = PrototypeSelection()
        ps.fit(self.dataset.X_train, self.dataset.y_train)
        X_prototypes, y_prototypes = ps.transform()

        reduction_ratio = ps.get_reduction_ratio()
        print(f"Исходный размер выборки: {len(self.dataset.X_train)}")
        print(f"Размер после отбора эталонов: {len(X_prototypes)}")
        print(f"Коэффициент сокращения: {reduction_ratio:.4f}")
        print()

        k_to_use = self.best_k if self.best_k is not None else (self.config.k_range[len(self.config.k_range) // 2] if self.config.k_range else 5)

        knn_prototypes = self.train_knn(X_prototypes, y_prototypes, k_to_use)
        metrics_prototypes = self.evaluate_knn(knn_prototypes, self.dataset.X_test, self.dataset.y_test, "KNN с отбором эталонов")

        if self.config.save_plots:
            save_path = os.path.join(self.config.plots_dir, 'prototype_selection.png')
            self.visualizer.plot_prototype_selection(
                self.dataset.X_train,
                self.dataset.y_train,
                X_prototypes,
                y_prototypes,
                save_path=save_path
            )

        if self.config.save_plots:
            y_pred_prototypes = knn_prototypes.predict(self.dataset.X_test)
            self.visualizer.plot_confusion_matrix(
                self.dataset.y_test,
                y_pred_prototypes,
                title='Матрица ошибок: KNN с отбором эталонов',
                save_path=os.path.join(self.config.plots_dir, 'confusion_matrix_prototypes.png')
            )

        self.prototype_metrics = {
            'metrics': metrics_prototypes,
            'reduction_ratio': reduction_ratio,
        }

    def run(self):
        self.load_dataset()
        self.find_best_k_with_loo()
        self.train_our()
        self.train_sklearn()
        self.prototype_selection_experiment()

        comparison_data = {
            'KNN': self.our_metrics,
            'Sklearn KNN': self.sklearn_metrics,
            'KNN (эталоны)': self.prototype_metrics['metrics']
        }

        if self.config.save_plots:
            save_path = os.path.join(self.config.plots_dir, 'comparison.png')
            self.visualizer.plot_comparison(comparison_data, save_path=save_path)

        print("Генерация отчета...")

        config_dict = {
            'dataset_name': self.config.dataset_name,
            'k_range': self.config.k_range
        }

        report_path = self.report_generator.generate_report(
            config=config_dict,
            loo_results=self.loo_results,
            our_metrics=self.our_metrics,
            sklearn_metrics=self.sklearn_metrics,
            prototype_metrics=self.prototype_metrics
        )

        print(f"Отчет сохранен: {report_path}")


if __name__ == "__main__":
    config = Config.from_yaml('config.yaml')
    runner = LabWorkRunner(config)
    runner.run()

