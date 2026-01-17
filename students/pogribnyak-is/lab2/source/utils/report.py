from typing import Dict, Optional
from datetime import datetime
import os

import numpy as np


class ReportGenerator:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
            self,
            config: Dict,
            loo_results: Optional[Dict] = None,
            our_metrics: Optional[Dict] = None,
            sklearn_metrics: Optional[Dict] = None,
            prototype_metrics: Optional[Dict] = None
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'report_{timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О ЛАБОРАТОРНОЙ РАБОТЕ №2\n")

            f.write("1. КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА\n")
            f.write(f"Датасет: {config.get('dataset_name', 'N/A')}\n")
            f.write(f"Диапазон k: {config.get('k_range', 'N/A')}\n\n")

            if loo_results:
                f.write("2. ПОДБОР ПАРАМЕТРА K МЕТОДОМ СКОЛЬЗЯЩЕГО КОНТРОЛЯ (LOO)\n")
                best_k = loo_results.get('best_k', 'N/A')
                best_error = loo_results.get('best_error', 'N/A')
                f.write(f"Лучшее значение k: {best_k}\n")
                f.write(f"Эмпирический риск при k={best_k}: {best_error:.4f}\n")
                f.write("Зависимость эмпирического риска от k:\n")
                for k, error in zip(loo_results.get('k_range', []), loo_results.get('errors', [])):
                    if isinstance(error, (int, float)): f.write(f"  k={k}: {error:.4f}\n")
                    else: f.write(f"  k={k}: N/A\n")
                f.write("\n")

            def write_metrics(title: str, metrics: Dict):
                f.write(f"{title}\n")
                if metrics:
                    f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n" if isinstance(metrics.get('accuracy'),
                                                                                              (int,
                                                                                               float)) else "Accuracy: N/A\n")
                    f.write(
                        f"Precision: {metrics.get('precision', 'N/A'):.4f}\n" if isinstance(metrics.get('precision'),
                                                                                            (int,
                                                                                             float)) else "Precision: N/A\n")
                    f.write(f"Recall: {metrics.get('recall', 'N/A'):.4f}\n" if isinstance(metrics.get('recall'), (int,
                                                                                                                  float)) else "Recall: N/A\n")
                    f.write(f"F1-score: {metrics.get('f1', 'N/A'):.4f}\n" if isinstance(metrics.get('f1'), (int,
                                                                                                            float)) else "F1-score: N/A\n")
                    if 'confusion_matrix' in metrics:
                        f.write("Confusion matrix:\n")
                        cm = metrics['confusion_matrix']
                        if isinstance(cm, (list, np.ndarray)):
                            for row in cm:
                                f.write("  " + " ".join(map(str, row)) + "\n")
                        else:
                            f.write(f"{cm}\n")
                else:
                    f.write("Метрики недоступны.\n")
                f.write("\n")

            if our_metrics: write_metrics("3. Метрики модели (полная выборка)", our_metrics)
            if prototype_metrics:
                reduction_ratio = prototype_metrics.get('reduction_ratio')
                f.write(f"Коэффициент сокращения выборки после отбора эталонов: {reduction_ratio:.4f}\n\n")
                write_metrics("4. Метрики модели (отобранные эталоны)", prototype_metrics.get('metrics', {}))
            if sklearn_metrics: write_metrics("5. Метрики Sklearn KNeighborsClassifier", sklearn_metrics)

        return report_path
