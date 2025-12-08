from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


class Visualizer:
    def __init__(self, epoch_train_loss: List[float], epoch_val_loss: List[float]):
        self.epoch_train_loss = epoch_train_loss
        self.epoch_val_loss = epoch_val_loss

    def plot_training_history(self, title: str = ""):
        plt.figure(figsize=(8, 6))
        epochs = range(len(self.epoch_train_loss))
        plt.plot(epochs, self.epoch_train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, self.epoch_val_loss, label="Validation Loss", linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{title}\nTrain and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        img_name = f"loss_{title.replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(os.path.join(IMAGES_DIR, img_name), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_margins(self, X_scores: np.ndarray, y: np.ndarray, title: str = ""):
        margins = X_scores * y
        margins_sorted = np.sort(margins)

        plt.figure(figsize=(10, 6))
        plt.plot(margins_sorted, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Samples')
        plt.ylabel('Margin')
        plt.title(f'Margins - {title}')
        plt.grid(True, alpha=0.3)

        img_name = f"margins_{title.replace(' ', '_')}.png"
        plt.savefig(os.path.join(IMAGES_DIR, img_name), dpi=300, bbox_inches='tight')
        plt.show()