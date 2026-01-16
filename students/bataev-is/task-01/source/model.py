import numpy as np
from nn_utils import activation_functions, derivative_functions, losses, d_losses
from numpy.lib.stride_tricks import sliding_window_view

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

class Layer:
    def __init__(self, n_neurons, activation: str):
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.activation = None
        self.d_activation = None
        self.save_inputs = False
        self.a = None
    
    def start_fit(self):
        self.save_inputs = True

    def stop_fit(self):
        self.a = None
        self.save_inputs = False
    
    def feedforward(self, x): pass
    def backward(self, delta, prevision_a, lr): pass
    def initialize_weights(self, size): pass

class Dense(Layer):
    def __init__(self, n_neurons, activation: str):
        if activation not in activation_functions:
            raise ValueError(f"Unknown activation: {activation}")
        super().__init__(n_neurons, activation)
        self.activation = activation_functions[activation]
        self.d_activation = derivative_functions[activation]

    def feedforward(self, x):
        # x.reshape(-1, 1) гарантирует вектор-столбец
        x_col = x.reshape(-1, 1)
        x_bias = np.vstack([np.array([[1.0]]), x_col]) # Добавляем bias вектор -> (n_features + 1, 1)
        z = self.weights.T @ x_bias
        if self.save_inputs:
            self.a = self.activation(z)
            return self.a.flatten()
        return self.activation(z).flatten()
    
    def backward(self, delta, prevision_a, lr): 
        delta = delta.reshape(-1, 1)

        if self.activation_name == 'softmax':
            pass # Delta уже правильная
        else:
            delta *= self.d_activation(self.a)

        # delta *= self.d_activation(self.a)
        
        prev_a_col = prevision_a.reshape(-1, 1)
        x_bias = np.vstack([np.array([[1.0]]), prev_a_col])
        grad = x_bias @ delta.T

        # Обновление весов
        self.weights -= grad * lr
        
        # Дельта для следующего слоя (без bias)
        next_delta = self.weights[1:,:] @ delta
        return next_delta
    
    def initialize_weights(self, input_size):
        # Xavier инициализация лучше для обучения, чем просто random
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.normal(0.0, scale, size=(input_size + 1, self.n_neurons))
        # self.weights = np.random.random(size=(input_size + 1, self.n_neurons))


class Flatten(Layer):
    def __init__(self):
        super().__init__(n_neurons=0, activation=None)
        self.input_shape = None

    def feedforward(self, x):
        self.input_shape = x.shape
        out = x.flatten()
        if self.save_inputs:
            self.a = out
        return out

    def backward(self, delta, prevision_a, lr):
        return delta.reshape(self.input_shape)


class Conv2D(Layer):
    def __init__(self, n_filters, filter_size: tuple, activation: str):
        if activation not in activation_functions:
            raise ValueError(f"Unknown activation: {activation}")
        super().__init__(n_filters, activation)
        self.filter_size = filter_size
        # В этой реализации n_neurons используем как площадь фильтра для удобства
        self.n_neurons = filter_size[0] * filter_size[1] 
        self.activation = activation_functions[activation]
        self.d_activation = derivative_functions[activation]

    def feedforward(self, x):
        window_shape = (self.filter_size[0], self.filter_size[1])
        windows = sliding_window_view(x, window_shape)
        windows = windows.reshape(-1, self.filter_size[0] * self.filter_size[1])
        
        out = windows @ self.weights + self.bias
        # Восстанавливаем размерность картинки
        out = out.reshape(x.shape[0] - self.filter_size[0] + 1, x.shape[1] - self.filter_size[1] + 1)
        
        if self.save_inputs:
            self.a = self.activation(out)
            return self.a
        return self.activation(out)

    def backward(self, delta, prevision_a, lr):
        delta *= self.d_activation(self.a)
        
        window_shape = (self.filter_size[0], self.filter_size[1])
        windows = sliding_window_view(prevision_a, window_shape)
        windows = windows.reshape(-1, self.filter_size[0] * self.filter_size[1])

        # Градиент весов
        grad = windows.T @ delta.flatten()
        
        # Обновление параметров
        self.weights -= grad.reshape(self.n_neurons, 1) * lr
        self.bias -= np.sum(delta) * lr

        # Расчет ошибки для предыдущего слоя (Full Convolution)
        w_reshaped = self.weights.reshape(self.filter_size)
        w_flipped = w_reshaped[::-1, ::-1] # Flip 180
        w_flipped_flat = w_flipped.reshape(-1, 1)
        
        pad_h = self.filter_size[0] - 1
        pad_w = self.filter_size[1] - 1
        delta_padded = np.pad(delta, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        delta_windows = sliding_window_view(delta_padded, window_shape)
        delta_windows = delta_windows.reshape(-1, self.filter_size[0] * self.filter_size[1])
        
        next_delta = delta_windows @ w_flipped_flat
        return next_delta.reshape(prevision_a.shape)

    def initialize_weights(self, input_size):
        # input_size здесь игнорируется для упрощения (считаем 1 канал входа)
        scale = np.sqrt(2.0 / (self.n_neurons))
        self.weights = np.random.normal(0.0, scale, size=(self.n_neurons, 1))
        # self.weights = np.random.random(size=(self.n_neurons,1))
        self.bias = np.random.random((1,))


class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        super().__init__(n_neurons=0, activation=None)
        self.pool_size = pool_size
        self.stride = stride

    def feedforward(self, x):
        h, w = x.shape
        # Обрезка для кратности
        h_new = h // self.pool_size * self.pool_size
        w_new = w // self.pool_size * self.pool_size
        x_cropped = x[:h_new, :w_new]
        
        x_reshaped = x_cropped.reshape(h_new // self.pool_size, self.pool_size,
                                       w_new // self.pool_size, self.pool_size)
        self.out = x_reshaped.max(axis=(1, 3))
        
        if self.save_inputs:
            self.a = self.out
        return self.out

    def backward(self, delta, prevision_a, lr):
        h, w = prevision_a.shape
        h_new = h // self.pool_size * self.pool_size
        w_new = w // self.pool_size * self.pool_size
        a_cropped = prevision_a[:h_new, :w_new]
        
        a_reshaped = a_cropped.reshape(h_new // self.pool_size, self.pool_size,
                                       w_new // self.pool_size, self.pool_size)
        
        delta_expanded = delta.repeat(self.pool_size, axis=0).repeat(self.pool_size, axis=1)
        delta_reshaped = delta_expanded.reshape(a_reshaped.shape)
        
        out_expanded = self.out.repeat(self.pool_size, axis=0).repeat(self.pool_size, axis=1)
        out_reshaped = out_expanded.reshape(a_reshaped.shape)
        
        mask = (a_reshaped == out_reshaped)
        d_input_reshaped = delta_reshaped * mask
        
        d_input = np.zeros_like(prevision_a)
        d_input[:h_new, :w_new] = d_input_reshaped.reshape(h_new, w_new)
        return d_input



class SimpleRNN(Layer):
    def __init__(self, n_neurons, input_shape, activation='tanh'):
        # input_shape = (seq_length, n_features)
        super().__init__(n_neurons, activation)
        self.seq_length = input_shape[0]
        self.n_features = input_shape[1]

        self.n_neurons = n_neurons
        
        # Инициализация весов
        # Wx: Вход -> Скрытое состояние
        self.Wx = np.random.randn(self.n_features, n_neurons) * 0.01
        # Wh: Скрытое -> Скрытое (Связь со своим прошлым)
        self.Wh = np.random.randn(n_neurons, n_neurons) * 0.01
        self.b = np.zeros((1, n_neurons))
        
    def feedforward(self, x):
        # x shape: (seq_length, n_features)
        self.inputs = x
        self.hs = { -1: np.zeros((1, self.n_neurons)) } # Словарь состояний h - каждый раз новый, так как берем на вход окно 
        
        # Проходим по всей временной последовательности - пока только tanh
        for t in range(self.seq_length):
            xt = x[t].reshape(1, -1)
            # h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)
            z = xt @ self.Wx + self.hs[t-1] @ self.Wh + self.b # shape (1, n_features) * (n_features, n_neurons) + (1, n_neurons) * (n_neurons, n_neurons) + (1, n_neurons) -> (1, n_neurons)
            self.hs[t] = np.tanh(z)
            
        # Возвращаем последнее скрытое состояние (Many-to-One)
        self.a = self.hs[self.seq_length - 1]
        return self.a.flatten() # shape (n_neurons,)

    def backward(self, delta, prevision_a, lr):
        # delta приходит от Dense слоя: shape (1, n_neurons)
        # Нам нужно протащить ошибку ОБРАТНО по времени (BPTT)
        
        delta = delta.reshape(1, -1)
        
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        
        dh_next = delta # Градиент, приходящий "из будущего" (или от следующего слоя)
        
        # Идем от последнего шага к первому (Каждый выход прошлого h по сути является входом следующего и Wx может обновляться от него)
        for t in reversed(range(self.seq_length)):

            # Идем по признакам текущего состояния - строка
            xt = self.inputs[t].reshape(-1, 1)
            
            # Градиент через tanh: dtanh = (1 - h^2) - пока только tanh
            d_z = dh_next * (1 - self.hs[t] ** 2)
            
            # Накапливаем градиенты весов
            dWx += xt @ d_z
            dWh += self.hs[t-1].T @ d_z
            db += d_z
            
            # Вычисляем ошибку для предыдущего шага времени (dh_prev)
            dh_next = d_z @ self.Wh.T
            
        # Обновляем веса (CLIP gradients, чтобы не взорвались!)
        for grad in [dWx, dWh, db]:
            np.clip(grad, -1, 1, out=grad)

        self.Wx -= dWx * lr
        self.Wh -= dWh * lr
        self.b -= db * lr
        
        # Мы не возвращаем ошибку для входа последовательности в этой лабе (обычно не нужно для первого слоя)
        return None 
    
    def initialize_weights(self, size):
        # Инициализация уже в __init__, заглушка
        pass

class GRU(Layer):
    def __init__(self, n_neurons, input_shape):
        super().__init__(n_neurons, activation='tanh')
        self.seq_length = input_shape[0]
        self.n_features = input_shape[1]

        self.n_neurons = n_neurons
        
        # Инициализация (W - для входа, U - для скрытого состояния)
        # Update gate (z)
        self.Wz = np.random.randn(self.n_features, n_neurons) * 0.01
        self.Uz = np.random.randn(n_neurons, n_neurons) * 0.01
        self.bz = np.zeros((1, n_neurons))
        
        # Reset gate (r)
        self.Wr = np.random.randn(self.n_features, n_neurons) * 0.01
        self.Ur = np.random.randn(n_neurons, n_neurons) * 0.01
        self.br = np.zeros((1, n_neurons))
        
        # Candidate hidden state (h_tilde)
        self.Wh = np.random.randn(self.n_features, n_neurons) * 0.01
        self.Uh = np.random.randn(n_neurons, n_neurons) * 0.01
        self.bh = np.zeros((1, n_neurons))

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def d_sigmoid(self, x): s = self.sigmoid(x); return s * (1 - s)
    def tanh(self, x): return np.tanh(x)
    def d_tanh(self, x): return 1 - np.tanh(x)**2

    def feedforward(self, x):
        self.inputs = x
        self.hs = { -1: np.zeros((1, self.n_neurons)) }
        self.zs = {} # store update gates
        self.rs = {} # store reset gates
        self.h_tildes = {} # store candidates
        
        for t in range(self.seq_length):
            xt = x[t].reshape(1, -1)
            h_prev = self.hs[t-1]
            
            # 1. Update gate
            z = self.sigmoid(xt @ self.Wz + h_prev @ self.Uz + self.bz)
            self.zs[t] = z
            
            # 2. Reset gate
            r = self.sigmoid(xt @ self.Wr + h_prev @ self.Ur + self.br)
            self.rs[t] = r
            
            # 3. Candidate h
            h_tilde = self.tanh(xt @ self.Wh + (r * h_prev) @ self.Uh + self.bh)
            self.h_tildes[t] = h_tilde
            
            # 4. Final h
            h = (1 - z) * h_prev + z * h_tilde
            self.hs[t] = h
            
        self.a = self.hs[self.seq_length - 1]
        return self.a.flatten()

    def backward(self, delta, prevision_a, lr):
        delta = delta.reshape(1, -1)
        dh_next = delta
        
        # Накопители градиентов
        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        
        for t in reversed(range(self.seq_length)):
            xt = self.inputs[t].reshape(1, -1)
            h_prev = self.hs[t-1]
            z = self.zs[t]
            r = self.rs[t]
            h_tilde = self.h_tildes[t]
            
            # Градиент потери по h[t]
            dh = dh_next
            
            # --- Разворачиваем формулы GRU назад ---
            # dL/dh_tilde = dh * z * (1 - tanh^2(...))
            dh_tilde = dh * z * (1 - h_tilde**2)
            
            # dL/dz = dh * (h_tilde - h_prev) * z(1-z)
            dz = dh * (h_tilde - h_prev) * z * (1 - z)
            
            # dL/dr (самое сложное, т.к. r внутри tanh через h_prev)
            # Chain rule: dL/dr = dL/dh_tilde * dh_tilde/d(r*h_prev) * ...
            # Упрощенно: dh_tilde протекает через Uh
            dr_term = (dh_tilde @ self.Uh.T) * h_prev
            dr = dr_term * r * (1 - r)
            
            # Накапливаем веса (Input weights)
            dWh += xt.T @ dh_tilde
            dWr += xt.T @ dr
            dWz += xt.T @ dz
            
            # Recurrent weights
            dUh += (r * h_prev).T @ dh_tilde
            dUr += h_prev.T @ dr
            dUz += h_prev.T @ dz
            
            # Biases
            dbh += dh_tilde
            dbr += dr
            dbz += dz
            
            # Вычисляем dh_prev (градиент для предыдущего шага)
            # Он складывается из путей через z, r и h_tilde
            dh_prev_z = dz @ self.Uz.T
            dh_prev_r = dr @ self.Ur.T
            dh_prev_h = (dh_tilde @ self.Uh.T) * r
            dh_prev_direct = dh * (1 - z) # Прямой проток мимо ворот
            
            dh_next = dh_prev_z + dh_prev_r + dh_prev_h + dh_prev_direct

        # Update weights (Clipping recommended)
        for grad in [dWh, dUh, dbh, dWz, dUz, dbz, dWr, dUr, dbr]:
            np.clip(grad, -1, 1, out=grad)
            
        self.Wz -= dWz * lr; self.Uz -= dUz * lr; self.bz -= dbz * lr
        self.Wr -= dWr * lr; self.Ur -= dUr * lr; self.br -= dbr * lr
        self.Wh -= dWh * lr; self.Uh -= dUh * lr; self.bh -= dbh * lr
        
        return None

class LSTM(Layer):
    def __init__(self, n_neurons, input_shape):
        super().__init__(n_neurons, activation='tanh')
        self.seq_length = input_shape[0]
        self.n_features = input_shape[1]

        self.n_neurons = n_neurons
        
        # Инициализация весов (Объединим Wx и Wh для компактности кода, но разделим по гейтам)
        # Order: Forget(f), Input(i), Cell(g), Output(o)
        self.Wf = np.random.randn(self.n_features, n_neurons) * 0.01; self.Uf = np.random.randn(n_neurons, n_neurons) * 0.01; self.bf = np.zeros((1, n_neurons))
        self.Wi = np.random.randn(self.n_features, n_neurons) * 0.01; self.Ui = np.random.randn(n_neurons, n_neurons) * 0.01; self.bi = np.zeros((1, n_neurons))
        self.Wg = np.random.randn(self.n_features, n_neurons) * 0.01; self.Ug = np.random.randn(n_neurons, n_neurons) * 0.01; self.bg = np.zeros((1, n_neurons))
        self.Wo = np.random.randn(self.n_features, n_neurons) * 0.01; self.Uo = np.random.randn(n_neurons, n_neurons) * 0.01; self.bo = np.zeros((1, n_neurons))

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def tanh(self, x): return np.tanh(x)

    def feedforward(self, x):
        self.inputs = x
        self.hs = { -1: np.zeros((1, self.n_neurons)) }
        self.cs = { -1: np.zeros((1, self.n_neurons)) } # Cell states
        # Cache for backprop
        self.cache = {} 
        
        for t in range(self.seq_length):
            xt = x[t].reshape(1, -1)
            h_prev = self.hs[t-1]
            c_prev = self.cs[t-1]
            
            # Gates
            f = self.sigmoid(xt @ self.Wf + h_prev @ self.Uf + self.bf)
            i = self.sigmoid(xt @ self.Wi + h_prev @ self.Ui + self.bi)
            g = self.tanh(   xt @ self.Wg + h_prev @ self.Ug + self.bg) # candidate cell
            o = self.sigmoid(xt @ self.Wo + h_prev @ self.Uo + self.bo)
            
            # States
            c = f * c_prev + i * g
            h = o * self.tanh(c)
            
            self.hs[t] = h
            self.cs[t] = c
            self.cache[t] = (f, i, g, o, self.tanh(c))
            
        self.a = self.hs[self.seq_length - 1]
        return self.a.flatten()

    def backward(self, delta, prevision_a, lr):
        delta = delta.reshape(1, -1)
        dh_next = delta
        dc_next = np.zeros_like(dh_next) # Градиент для Cell state
        
        # Инициализация градиентов весов нулями (аналогично GRU, опустим для краткости создание переменных)
        dWf = np.zeros_like(self.Wf); dUf = np.zeros_like(self.Uf); dbf = np.zeros_like(self.bf)
        dWi = np.zeros_like(self.Wi); dUi = np.zeros_like(self.Ui); dbi = np.zeros_like(self.bi)
        dWg = np.zeros_like(self.Wg); dUg = np.zeros_like(self.Ug); dbg = np.zeros_like(self.bg)
        dWo = np.zeros_like(self.Wo); dUo = np.zeros_like(self.Uo); dbo = np.zeros_like(self.bo)

        for t in reversed(range(self.seq_length)):
            xt = self.inputs[t].reshape(1, -1)
            h_prev = self.hs[t-1]
            c_prev = self.cs[t-1]
            f, i, g, o, tanh_c = self.cache[t]
            
            # Градиент приходит в h, потом идет в c
            dh = dh_next
            # dL/do = dL/dh * tanh(c) * o(1-o)
            do = dh * tanh_c * o * (1 - o)
            
            # Градиент для C складывается из "будущего dc" и "текущего dh"
            dc = dc_next + (dh * o * (1 - tanh_c**2))
            
            # Гейты C
            dg = dc * i * (1 - g**2) # tanh derivative
            di = dc * g * i * (1 - i) # sigmoid derivative
            df = dc * c_prev * f * (1 - f) # sigmoid derivative
            
            # Накапливаем веса (пример для f, остальные аналогично)
            dWf += xt.T @ df; dUf += h_prev.T @ df; dbf += df
            dWi += xt.T @ di; dUi += h_prev.T @ di; dbi += di
            dWg += xt.T @ dg; dUg += h_prev.T @ dg; dbg += dg
            dWo += xt.T @ do; dUo += h_prev.T @ do; dbo += do
            
            # Считаем dh_prev и dc_next
            dc_next = dc * f
            dh_next = df @ self.Uf.T + di @ self.Ui.T + dg @ self.Ug.T + do @ self.Uo.T

        # Update (with clip)
        for grad in [dWf, dUf, dbf, dWi, dUi, dbi, dWg, dUg, dbg, dWo, dUo, dbo]:
            np.clip(grad, -1, 1, out=grad)
            
        self.Wf -= dWf * lr; self.Uf -= dUf * lr; self.bf -= dbf * lr
        self.Wi -= dWi * lr; self.Ui -= dUi * lr; self.bi -= dbi * lr
        self.Wg -= dWg * lr; self.Ug -= dUg * lr; self.bg -= dbg * lr
        self.Wo -= dWo * lr; self.Uo -= dUo * lr; self.bo -= dbo * lr
        
        return None



class Input(Layer):
    def __init__(self, n_features: int):
        '''
        Входной слой, где n_features - количество признаков на входе
        '''
        super().__init__(n_features, None)


class Input2D(Layer):
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        super().__init__(input_shape[0] * input_shape[1], None)

class Input3D(Layer):
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        super().__init__(input_shape[0] * input_shape[1] * input_shape[2], None)


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: str = 'mse'):
        self.layers = layers
        self.count_layers = len(layers)
        self.activate_loss = losses[loss]
        self.deactivate_loss = d_losses[loss]
        self.loss_name = loss

        # Логика отслеживания размерностей для инициализации весов
        current_input_shape = None
        current_n_neurons = 0
        
        if isinstance(self.layers[0], Input2D):
             current_input_shape = self.layers[0].input_shape
             current_n_neurons = self.layers[0].n_neurons
        else:
             # Если вдруг Input обычный
             current_n_neurons = self.layers[0].n_neurons

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            
            if isinstance(layer, Conv2D):
                 layer.initialize_weights(current_n_neurons)
                 h, w = current_input_shape
                 out_h = h - layer.filter_size[0] + 1
                 out_w = w - layer.filter_size[1] + 1
                 current_input_shape = (out_h, out_w)
                 # ВАЖНО: кол-во нейронов выхода = пиксели * 1 фильтр (в этой реализации)
                 current_n_neurons = out_h * out_w 
                 
            elif isinstance(layer, MaxPooling2D):
                 h, w = current_input_shape
                 out_h = h // layer.pool_size
                 out_w = w // layer.pool_size
                 current_input_shape = (out_h, out_w)
                 current_n_neurons = out_h * out_w
                 
            elif isinstance(layer, Flatten):
                 current_n_neurons = current_input_shape[0] * current_input_shape[1]
            
            elif isinstance(layer, SimpleRNN) or isinstance(layer, GRU) or isinstance(layer, LSTM):
                current_n_neurons = layer.n_neurons
        
            elif isinstance(layer, Dense):
                 layer.initialize_weights(current_n_neurons)
                 current_n_neurons = layer.n_neurons

    def feedforward(self, x):
        a = x
        for layer in self.layers[1:]:
            a = layer.feedforward(a)
        return a
        
    def feedforward_batch(self, X):
        preds = []
        for x in X:
            preds.append(self.feedforward(x))
        return np.array(preds)

    def _train_one_sample(self, x, y, lr):
        # 1. Forward
        a = x
        self.layers[0].a = x
        for layer in self.layers[1:]:
            a = layer.feedforward(a)
            
        # 2. Loss
        y_col = y.reshape(-1, 1)
        a_col = a.reshape(-1, 1)

        if self.loss_name == 'categorical_cross_entropy' and \
           self.layers[-1].activation_name == 'softmax':
            
            # формула: градиент по входу в softmax равен (Pred - True)
            delta = a_col - y_col 
            
        else:
            # Стандартный случай (например MSE + Sigmoid)
            delta = self.deactivate_loss(y_col, a_col)

        # delta = self.deactivate_loss(y_col, a_col)

        # 3. Backward
        for i in range(1, self.count_layers):
            current_layer = self.layers[-i]
            prev_layer_a = self.layers[-i-1].a
            delta = current_layer.backward(delta, prev_layer_a, lr)

    def fit(self, X, y, epochs=10, lr=0.01):
        self.start_fit_init()
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            it = zip(X_shuffled, y_shuffled)
            if tqdm is not None:
                it = tqdm(it, total=len(X_shuffled), desc=f"Epoch {epoch+1}")
            for x_i, y_i in it:
                self._train_one_sample(x_i, y_i, lr)
            
            print(f'\nОшибка {self.loss_name}:', self.activate_loss(y_shuffled[0], self.feedforward(X_shuffled[0])))
        self.stop_fit_init()

    def start_fit_init(self):
        for layer in self.layers: layer.start_fit()
    def stop_fit_init(self):
        for layer in self.layers: layer.stop_fit()


class LinearClassifier:
    """
    Multiclass linear classifier (scores = X @ W + b) trained with MSE loss.
    Implemented in pure numpy to satisfy lab-01 requirements:
    - margin computation
    - SGD with momentum
    - L2 regularization
    - steepest (full-batch) gradient descent with backtracking line search
    - sample presentation by |margin| (hard samples first)
    """

    def __init__(self, n_features: int, n_classes: int, seed: int | None = None):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.rng = np.random.default_rng(seed)
        # W includes bias as the first row: shape (n_features + 1, n_classes)
        self.W = self.rng.normal(0.0, 0.01, size=(self.n_features + 1, self.n_classes))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def scores(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return Xb @ self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(X), axis=1)

    def margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Multiclass margin: s_true - max_{j != y} s_j
        Positive margin => correctly classified with margin.
        """
        S = self.scores(X)  # (n, C)
        y = np.asarray(y).astype(int)
        n = S.shape[0]
        s_true = S[np.arange(n), y]
        S_masked = S.copy()
        S_masked[np.arange(n), y] = -np.inf
        s_other = np.max(S_masked, axis=1)
        return s_true - s_other

    def loss_mse(self, X: np.ndarray, Y_onehot: np.ndarray, l2_lambda: float = 0.0) -> float:
        S = self.scores(X)
        diff = (S - Y_onehot)
        data_loss = 0.5 * np.mean(np.sum(diff * diff, axis=1))
        reg = 0.5 * float(l2_lambda) * float(np.sum(self.W[1:, :] ** 2))
        return float(data_loss + reg)

    def grad_mse(self, X: np.ndarray, Y_onehot: np.ndarray, l2_lambda: float = 0.0) -> np.ndarray:
        Xb = self._add_bias(X)  # (n, d+1)
        S = Xb @ self.W
        n = Xb.shape[0]
        grad = (Xb.T @ (S - Y_onehot)) / n
        if l2_lambda != 0.0:
            reg = np.zeros_like(self.W)
            reg[1:, :] = self.W[1:, :]
            grad = grad + float(l2_lambda) * reg
        return grad

    def init_correlation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Practical "correlation-like" init via class centroids.
        Scores become proportional to negative squared distance to class mean.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        W = np.zeros((self.n_features + 1, self.n_classes), dtype=float)
        for c in range(self.n_classes):
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            W[1:, c] = mu
            W[0, c] = -0.5 * float(mu @ mu)
        self.W = W

    def fit_sgd_momentum(
        self,
        X: np.ndarray,
        Y_onehot: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 50,
        lr: float = 0.05,
        momentum: float = 0.9,
        l2_lambda: float = 0.0,
        quality_alpha: float = 0.02,
        order: str = "random",  # random | margin_abs
        seed: int | None = None,
    ) -> dict:
        """
        Returns history dict with recurrent quality estimate and epoch metrics.
        """
        X = np.asarray(X)
        Y_onehot = np.asarray(Y_onehot)
        y_labels = np.asarray(y_labels).astype(int)

        rng = np.random.default_rng(seed)
        v = np.zeros_like(self.W)
        q = None
        history = {
            "q": [],
            "train_loss": [],
            "train_acc": [],
        }

        for _epoch in range(int(epochs)):
            if order == "random":
                idx = rng.permutation(X.shape[0])
            elif order == "margin_abs":
                m = self.margin(X, y_labels)
                idx = np.argsort(np.abs(m))  # hardest first
            else:
                raise ValueError(f"Unknown order='{order}'. Use 'random' or 'margin_abs'.")

            it = idx
            if tqdm is not None:
                it = tqdm(it, total=len(idx), desc=f"SGD epoch {_epoch+1} ({order})")
            for i in it:
                xi = X[i : i + 1, :]
                yi = Y_onehot[i : i + 1, :]

                Xb = self._add_bias(xi)  # (1, d+1)
                si = Xb @ self.W  # (1, C)
                diff = (si - yi)  # (1, C)

                # per-sample loss (no reg) for recurrent estimate
                li = 0.5 * float(np.sum(diff * diff))
                q = li if q is None else (1.0 - quality_alpha) * q + quality_alpha * li
                history["q"].append(float(q))

                grad = Xb.T @ diff  # (d+1, C)
                if l2_lambda != 0.0:
                    reg = np.zeros_like(self.W)
                    reg[1:, :] = self.W[1:, :]
                    grad = grad + float(l2_lambda) * reg

                v = momentum * v - lr * grad
                self.W = self.W + v

            history["train_loss"].append(self.loss_mse(X, Y_onehot, l2_lambda=l2_lambda))
            history["train_acc"].append(float(np.mean(self.predict(X) == y_labels)))

        return history

    def fit_steepest_descent(
        self,
        X: np.ndarray,
        Y_onehot: np.ndarray,
        max_iters: int = 200,
        lr0: float = 1.0,
        l2_lambda: float = 0.0,
        armijo_c: float = 1e-4,
        backtracking_beta: float = 0.5,
        max_ls_iters: int = 40,
    ) -> dict:
        """
        Full-batch steepest descent with Armijo backtracking line search.
        """
        history = {"loss": []}
        for _ in range(int(max_iters)):
            loss0 = self.loss_mse(X, Y_onehot, l2_lambda=l2_lambda)
            history["loss"].append(loss0)
            grad = self.grad_mse(X, Y_onehot, l2_lambda=l2_lambda)
            g2 = float(np.sum(grad * grad))
            if g2 < 1e-12:
                break

            t = float(lr0)
            for _ls in range(int(max_ls_iters)):
                W_old = self.W
                self.W = W_old - t * grad
                loss_new = self.loss_mse(X, Y_onehot, l2_lambda=l2_lambda)
                if loss_new <= loss0 - armijo_c * t * g2:
                    break
                self.W = W_old
                t *= float(backtracking_beta)
            else:
                # failed line search; stop
                break

        return history