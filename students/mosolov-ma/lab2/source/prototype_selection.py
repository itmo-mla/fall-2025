import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class PrototypeSelector:
    def __init__(self, X, y, k=5, tol=1e-2, max_extra_steps=10, model='custom', split_method='custom'):
        self.X = X
        self.y = y
        self.k = k
        self.tol = tol
        self.max_extra_steps = max_extra_steps
        self.model = model
        self.split_method = split_method
        self.error_history = []
        self.size_history = []
        self.prototype_indices = None

    def _compute_ccv_error(self, omega_idx):
        if len(omega_idx) <= 1:
            return 1.0

        X_omega = self.X[omega_idx]
        y_omega = self.y[omega_idx]
        n = len(omega_idx)
        errors = 0

        for i in range(n):
            # Обучаем на Ω без i-го объекта
            X_train = np.delete(X_omega, i, axis=0)
            y_train = np.delete(y_omega, i, axis=0)
            x_test = X_omega[i].reshape(1, -1)

            if self.model == 'custom':
                model = KNN(k=self.k)
            else:  # sklearn
                model = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')

            model.fit(X_train, y_train)
            y_pred = model.predict(x_test)[0]
            if y_pred != y_omega[i]:
                errors += 1

        return errors / n
    

    def _compute_full_loo_error_correct(self, omega_idx):
        if len(omega_idx) <= 20:
            return 1.0
        
        total_errors = 0
        n = len(self.X)

        # Предварительно обучим базовую модель на Ω (для объектов не из Ω)
        X_omega = self.X[omega_idx]
        y_omega = self.y[omega_idx]

        for i in range(n):
            x_test = self.X[i].reshape(1, -1)
            y_true = self.y[i]

            if i in omega_idx:
                # x_i — эталон → исключаем его из обучающей выборки
                mask = np.array(omega_idx) != i
                if mask.sum() == 0:
                    # Нечего обучать — ошибка = 1
                    total_errors += 1
                    continue
                X_train = X_omega[mask]
                y_train = y_omega[mask]
            else:
                # x_i — не эталон → обучаем на всём Ω
                X_train = X_omega
                y_train = y_omega

            if self.model == 'custom':
                model = KNN(k=self.k)
            else:
                model = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')

            model.fit(X_train, y_train)
            y_pred = model.predict(x_test)[0]
            if y_pred != y_true:
                total_errors += 1

        return total_errors / n

    def find_prototypes(self):
        omega_idx = list(range(len(self.X)))
        current_error = self._compute_full_loo_error_correct(omega_idx)

        self.error_history = [current_error]
        self.size_history = [len(omega_idx)]

        print(f"[Старт] LOO на полной выборке: {current_error:.4f}, |Ω| = {len(omega_idx)}")

        iteration = 0
        extra_steps_done = 0
        strong_jump_occurred = False
        min_size = max(self.k, 2)

        while len(omega_idx) > min_size:
            best_error = float('inf')
            best_to_remove = None

            for i, idx in enumerate(omega_idx):
                candidate_omega = omega_idx[:i] + omega_idx[i+1:]
                if len(candidate_omega) < min_size:
                    continue
                error = self._compute_full_loo_error_correct(candidate_omega)
                if error < best_error:
                    best_error = error
                    best_to_remove = i

            if best_to_remove is None:
                break

            if not strong_jump_occurred:
                if best_error > current_error + self.tol and (best_error - current_error) > 0.02:
                    strong_jump_occurred = True
                    print(f"[!] Сильный скачок ошибки на итерации {iteration + 1}: "
                          f"{current_error:.4f} → {best_error:.4f}")

            removed_idx = omega_idx.pop(best_to_remove)
            current_error = best_error
            iteration += 1

            self.error_history.append(current_error)
            self.size_history.append(len(omega_idx))

            print(f"Итерация {iteration}: удалён объект {removed_idx}, "
                  f"ошибка на Ω = {current_error:.4f}, |Ω| = {len(omega_idx)}")

            if strong_jump_occurred:
                extra_steps_done += 1
                if extra_steps_done >= self.max_extra_steps:
                    print(f"Завершено: выполнено {self.max_extra_steps} шагов после сильного скачка.")
                    break

        self.prototype_indices = omega_idx
        return np.array(omega_idx)

    def plot_error_history(self, title="LOO Error on Full Dataset During Prototype Selection"):
        if not self.error_history:
            raise ValueError("Сначала вызовите find_prototypes()!")
        plt.figure(figsize=(10, 6))
        plt.plot(self.size_history, self.error_history, marker='o', linestyle='-', color='b')
        plt.gca().invert_xaxis()  # Меньше эталонов → вправо
        plt.xlabel("Число эталонов |Ω|")
        plt.ylabel("LOO Error (на полной выборке)")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _apply_pca_if_needed(self, X):
        """
        Преобразует данные в 2D с помощью PCA, если нужно.
        Возвращает (X_2d, pca_object), где X_2d имеет shape (n_samples, 2).
        """
        if X.shape[1] == 2:
            return X, None  # PCA не нужен
        elif X.shape[1] < 2:
            raise ValueError("Данные должны иметь хотя бы 2 признака.")
        else:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            return X_2d, pca


    def plot_decision_boundary(self, resolution=100, figsize=(8, 6)):
        if self.prototype_indices is None:
            raise ValueError("Сначала вызовите find_prototypes()!")

        # Применяем PCA ко всем данным
        X_2d, pca = self._apply_pca_if_needed(self.X)
        if pca is not None:
            X_proto_2d = pca.transform(self.X[self.prototype_indices])
        else:
            X_proto_2d = self.X[self.prototype_indices]

        classes = np.unique(self.y)
        n_classes = len(classes)

        # # Границы сетки
        # x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        # y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
        #                      np.linspace(y_min, y_max, resolution))

        # Цветовые палитры
        cmap_light = plt.cm.get_cmap('Pastel1', n_classes)
        cmap_bold = plt.cm.get_cmap('Set1', n_classes)

        plt.figure(figsize=figsize)
        # plt.contourf(xx, yy, cmap=cmap_light, alpha=0.8)

        # Наносим ВСЕ точки с их истинными метками
        for i, cls in enumerate(classes):
            mask = self.y == cls
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[cmap_bold(i)], label=f'Класс {cls}', edgecolor='k', s=20)

        # Выделяем эталоны — контурными кружками
        plt.scatter(X_proto_2d[:, 0], X_proto_2d[:, 1],
                    c='none', edgecolor='white', linewidth=2, s=80,
                    label='Эталоны', marker='o')

        title = f'Граница решения kNN (k={self.k})'
        if pca is not None:
            title += " (PCA 2D)"
        title += f'\nна {len(self.prototype_indices)} эталонах'
        plt.title(title)
        plt.xlabel('Первая главная компонента' if pca else 'Признак 1')
        plt.ylabel('Вторая главная компонента' if pca else 'Признак 2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()