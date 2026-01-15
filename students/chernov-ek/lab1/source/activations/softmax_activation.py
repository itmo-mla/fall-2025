import numpy as np

from source.activations import ABCActivation


class SoftmaxActivation(ABCActivation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs - np.max(inputs))
        self.A = exp/np.sum(exp)
        return self.A
    
    def partial_derivative_wrt_z(self, i_neuron: int) -> float:
        # Матрица Якоби
        jacobian = np.diagflat(self.A) - np.outer(self.A, self.A)
        return jacobian


# Temp
# # Параметры
# np.random.seed(42)
# num_inputs = 5
# num_classes = 3
# learning_rate = 0.1
# num_samples = 10

# # Входы и метки
# X = np.random.randn(num_samples, num_inputs)
# y = np.random.randint(0, num_classes, size=num_samples)

# # Инициализация весов и смещений
# W = np.random.randn(num_inputs, num_classes)
# b = np.zeros(num_classes)

# # Softmax
# def softmax(z):
#     exp_z = np.exp(z - np.max(z))
#     return exp_z / np.sum(exp_z)

# # One-hot encoding
# Y_one_hot = np.zeros((num_samples, num_classes))
# for n in range(num_samples):
#     Y_one_hot[n, y[n]] = 1

# # Поэлементное обновление весов
# for n in range(num_samples):
#     # Вычисляем z и a для примера n
#     z = np.zeros(num_classes)
#     for j in range(num_classes):
#         for i in range(num_inputs):
#             z[j] += X[n, i] * W[i, j]
#         z[j] += b[j]
    
#     a = softmax(z)
    
#     # dL/da = a - y_one_hot (для кросс-энтропии, но оставляем явно)
#     dL_da = a - Y_one_hot[n]
    
#     # Для каждого нейрона вычисляем da/dz в виде вектора
#     for j in range(num_classes):          # нейрон
#         da_dz = np.zeros(num_classes)
#         for k in range(num_classes):
#             if j == k:
#                 da_dz[k] = a[j] * (1 - a[j])
#             else:
#                 da_dz[k] = -a[j] * a[k]
        
#         print(da_dz)
#         # dL/dz_j = sum_k (dL/da_k * da_k/dz_j)
#         dL_dz = 0
#         for k in range(num_classes):
#             dL_dz += dL_da[k] * da_dz[k]
        
#         # Обновление весов по входам
#         for i in range(num_inputs):
#             dz_dw = X[n, i]                   # dz_j/dW_ij = x_i
#             dL_dW = dL_dz * dz_dw
#             W[i, j] -= learning_rate * dL_dW
        
#         # Обновление смещения
#         b[j] -= learning_rate * dL_dz
        
#         break
#     break

# # print("Обновленные веса:\n", W)
# # print("Обновленные смещения:\n", b)


# jacobian = np.zeros((3, 3))
# for i in range(3):
#     for j in range(3):
#         if i == j:
#             jacobian[i, j] = outputs[i] * (1 - outputs[j])
#         else:
#             jacobian[i, j] = -outputs[i] * outputs[j]

# jacobian
