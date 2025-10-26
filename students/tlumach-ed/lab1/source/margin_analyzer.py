import os
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

class Margin:
    def __init__(self, output_dir: str = IMAGES_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_margins(self, scores: np.ndarray, y: np.ndarray) -> np.ndarray:
        # scores: (n,), y: (n,)
        return scores * y

    def plot_margins(self, X_scores: np.ndarray, y: np.ndarray, title: str = "") -> np.ndarray:
        margins = self.compute_margins(X_scores, y)
        margins_sorted = np.sort(margins)

        plt.figure(figsize=(10, 6))
        plt.plot(margins_sorted, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Образцы (ранжированные)')
        plt.ylabel('Отступ')
        plt.title(f'Ранжированные отступы объектов - {title}')
        plt.grid(True, alpha=0.3)

        noise_bound = -0.02
        trusted_bound = 0.1
        noise_end = np.searchsorted(margins_sorted, noise_bound, side='right')
        trusted_start = np.searchsorted(margins_sorted, trusted_bound, side='right')

        plt.axvline(x=noise_end, color='orange', linestyle='--', alpha=0.7, label='Граница шума')
        plt.axvline(x=trusted_start, color='green', linestyle='--', alpha=0.7, label='Граница доверия')
        plt.legend()

        img_name = f"margins_{title.replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, img_name), dpi=300, bbox_inches='tight')
        plt.show()

        return margins_sorted
