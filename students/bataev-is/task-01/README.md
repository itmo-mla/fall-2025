# Лабораторная работа №1 — Линейная классификация (Iris)

### 1) Датасет

* **Источник**: `source/iris.csv` (классический Iris, 150 объектов).
* **Признаки**: 4 числовых (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
* **Классы**: 3 (`setosa`, `versicolor`, `virginica`).
* **Предобработка**:
  * Стратифицированный split: **train/val/test = 60/20/20**
  * Стандартизация по train: $x' = \frac{x - \mu}{\sigma}$

Код загрузки/предобработки: `source/main.py`.

**Примечание по инструментам**:
* Анализ и загрузка датасета выполняются через **pandas**.
* Вся логика модели/обучения (градиенты, оптимизация) реализована на **numpy**.

---

### 2) Отступ (margin) объекта

Для многоклассового линейного классификатора со score-функциями $s_k(x)$ определим отступ:

$$
m(x, y) = s_{y}(x) - \max_{j \neq y} s_j(x)
$$

* $m > 0$ — объект классифицируется правильно с запасом.
* $m < 0$ — объект ошибочно классифицирован (самый сильный конкурент «победил»).

В `source/main.py` строится гистограмма отступов **до** и **после** обучения:
* `margin_hist_before.png`
* `margin_hist_after.png`

---

### 3) Градиент функции потерь

Используется **квадратичная функция потерь (MSE)** на one-hot разметке:

$$
L(W) = \frac{1}{2N} \sum_{i=1}^{N} \lVert s(x_i) - y_i \rVert^2 + \frac{\lambda}{2}\lVert W_{\text{no-bias}}\rVert^2
$$

Где:
* $s(x) = [1, x]^T W$ — линейные скоры (bias включён как первая компонента).
* $y_i$ — one-hot вектор класса.
* $\lambda$ — коэффициент L2-регуляризации.
* Bias-строка не регуляризуется.

Градиент:

$$
\nabla_W L = \frac{1}{N} X_b^T (X_b W - Y) + \lambda \cdot \tilde{W}
$$

Где $X_b = [\mathbf{1}, X]$, а $\tilde{W}$ — это $W$ с занулённой первой строкой (bias).

---

### 4) Рекуррентная оценка функционала качества

В процессе SGD ведётся рекуррентная (экспоненциальная) оценка:

$$
Q_t = (1-\alpha)Q_{t-1} + \alpha \cdot \ell_t
$$

Где $\ell_t$ — текущая (per-sample) квадратичная ошибка, $\alpha \in (0,1)$.

---

### 5) SGD с инерцией (momentum)

Для шага SGD по одному объекту:

$$
v \leftarrow \mu v - \eta \nabla_W \ell
$$
$$
W \leftarrow W + v
$$

Где $\mu$ — momentum, $\eta$ — learning rate.

---

### 6) L2 регуляризация

Добавляется штраф $\frac{\lambda}{2}\lVert W_{\text{no-bias}}\rVert^2$, что эквивалентно добавлению в градиент члена $\lambda W$ (для всех строк, кроме bias).

---

### 7) Скорейший градиентный спуск

Реализован **full-batch** steepest descent с линейным поиском шага (Armijo backtracking):
* Стартовый шаг `lr0`.
* Пока условие Армихо не выполнено — уменьшаем шаг в `beta` раз.

---

### 8) Предъявление объектов «по модулю отступа»

Вариант предъявления:
* В начале эпохи считаем $m(x_i, y_i)$ для всех train-объектов.
* Сортируем по $|m|$ по возрастанию (самые «трудные» — первыми).
* Выполняем SGD в этом порядке.

---

### 9) Обучение (3 режима)

В `source/main.py` автоматически запускаются:
1. **Инициализация через корреляцию (centroid-init)** + SGD(momentum) + L2.
2. **Случайная инициализация + мультистарт** (выбор лучшего по val).
3. **Случайная инициализация + предъявление по |margin|**.

---

### 10) Оценка качества

Считаются:
* Accuracy на train/val/test.
* Confusion matrix на test для лучшей модели.

---

### 11) Сравнение со sklearn (эталонные модели)

В `source/main.py` добавлено опциональное сравнение с реализациями из **scikit-learn** (на тех же train/val/test и после той же стандартизации по train):

* `RidgeClassifier` — ближайший аналог **least-squares / ridge** подхода (по смыслу ближе всего к MSE+L2).
* `LogisticRegression` — сильный линейный baseline.
* `LinearSVC` — линейная SVM как популярный baseline.

Включение/выключение: `--sklearn / --no-sklearn` (по умолчанию включено).

При включённом сравнении скрипт печатает `acc` для **train/val/test** для каждой baseline-модели.

---

### 12) Как запустить

**Установить зависимости:**
> pip install -U pandas tqdm matplotlib scikit-learn numpy

**Запуск со стандартными параметрами:**
> python3 source/main.py

**Запуск с кастомными параметрами:**
> python3 source/main.py --epochs 120 --lr 0.03 --l2 0.001 --momentum 0.9 --multistart 25

**Запуск со sklearn-сравнением / без него:**
> python3 source/main.py --sklearn
>
> python3 source/main.py --no-sklearn

---

### Результаты (пример вывода)

Параметры прогона: **seed=42**, **train/val/test=60/20/20**, стандартизация по train, **epochs=100**, **lr=0.05**, **momentum=0.2**, **l2=0.001**, **multistart=15**.

#### Сравнительная таблица — ручная реализация (numpy)

| Метод (manual) | Train acc | Val acc | Test acc | Macro-F1 (test) |
|---|---:|---:|---:|---:|
| `corr_init_sgd` | 0.8000 | 0.9333 | 0.9000 | 0.8997 |
| `multistart_sgd` | 0.8111 | 1.0000 | 0.8333 | 0.8329 |
| `margin_order_sgd` | 0.6667 | 0.6667 | 0.6667 | 0.5473 |
| `steepest_descent` | 0.8000 | 0.9667 | 0.9000 | 0.8997 |

#### Сравнительная таблица — sklearn baselines

| Модель (sklearn) | Train acc | Val acc | Test acc | Macro-F1 (test) |
|---|---:|---:|---:|---:|
| `RidgeClassifier(alpha=1.0)` | 0.8000 | 0.9667 | 0.9000 | 0.8997 |
| `LogisticRegression` | 0.9556 | 1.0000 | 1.0000 | 1.0000 |
| `LinearSVC(C=1.0)` | 0.9556 | 1.0000 | 0.9667 | 0.9666 |

#### Детали (test) — confusion matrix / classification report

Для подробных метрик sklearn (confusion matrix + precision/recall/F1) см. вывод `source/main.py` при запуске с `--sklearn`.
