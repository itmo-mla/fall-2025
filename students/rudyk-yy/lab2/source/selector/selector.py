import numpy as np
from .compactness import Compactness
import matplotlib.pyplot as plt

class Selector:
    def __init__(self, X, y, epsilon=1e-6, verbose=False, k=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.epsilon = epsilon
        self.verbose = verbose
        self.k = k
        self.best_ccv = np.inf
        self.history = []

    def select_remove(self):
        C = Compactness(self.X, self.y)
        omega = list(range(len(self.X)))  
        improved = True

        print(self.X.shape)
        while improved and len(omega) > 1:
            improved = False
            best_i = []
            best_ccv = self.best_ccv
            print(self.X.shape[1])
            for i in omega:
                omega_new = [j for j in omega if j != i]
                ccv_new = C.CCV(np.array(omega_new),k=self.k, l=self.X.shape[1])
                print(f"\rПробуем удалить {i}: CCV={ccv_new:.6f}", end="")
                if ccv_new < best_ccv - self.epsilon:
                    best_ccv = ccv_new
                    best_i.append(i)
                
            print(len(best_i))
            if len(best_i) > 0:
                print(len(best_i))
                for i in best_i:
                    omega.remove(i)
                    if self.verbose:
                     print(f"Удалили {i}, осталось {len(omega)}, CCV={best_ccv:.6f}")
                self.best_ccv = best_ccv
                improved = True
                self.history.append((len(omega), best_ccv))
                

        return np.array(omega)
    def select_add(self):
        C = Compactness(self.X, self.y)
        classes = np.unique(self.y)

        Omega = [np.where(self.y == c)[0][0] for c in classes]
        Omega = list(set(Omega))

        self.best_ccv = np.inf
        if self.verbose:
            print(f"Начальное CCV={self.best_ccv:.6f}, Ω={Omega}")

        improved = True
        while improved:
            improved = False
            best_ccv = self.best_ccv
            best_i = None

            for i in range(len(self.X)):
                if i in Omega:
                    continue

                omega_new = Omega + [i]
                ccv_new = C.CCV(np.array(omega_new), k=self.k, l=self.X.shape[1])


                if ccv_new < best_ccv - self.epsilon:
                    best_ccv = ccv_new
                    best_i = i

            if best_i is not None:
                Omega.append(best_i)
                self.best_ccv = best_ccv
                improved = True
                self.history.append((len(Omega), best_ccv))
                if self.verbose:
                    print(f"\nДобавили {best_i}, CCV={best_ccv:.6f}, размер Ω={len(Omega)}")
        return np.array(Omega)

    def plot_history(self):
        if not self.history:
            print("Нет истории — сначала вызови select()")
            return
        sizes, ccv_values = zip(*self.history)
        plt.plot(sizes, ccv_values, marker='o')
        plt.xlabel("Размер множества Ω")
        plt.ylabel("CCV(Ω)")
        plt.title("Изменение CCV при жадном отборе эталонов")
        plt.gca().invert_xaxis()
        plt.show()
