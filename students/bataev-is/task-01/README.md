## Лабораторная работа №1 — Линейная классификация (Iris)

### 1) Датасет

- **Источник**: `source/iris.csv` (классический Iris, 150 объектов).
- **Признаки**: 4 числовых (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
- **Классы**: 3 (`setosa`, `versicolor`, `virginica`).
- **Предобработка**:
  - стратифицированный split: **train/val/test = 60/20/20**
  - стандартизация по train: \(x' = \frac{x - \mu}{\sigma}\)

Код загрузки/предобработки: `source/main.py`.

**Примечание по инструментам**:
- анализ и загрузка датасета выполняются через **pandas**
- вся логика модели/обучения (градиенты, оптимизация) реализована на **numpy**

---

### 2) Отступ (margin) объекта

Для многоклассового линейного классификатора со score-функциями \(s_k(x)\) определим отступ:

\[
m(x, y) = s_{y}(x) - \max_{j \ne y} s_j(x)
\]

- \(m>0\) — объект классифицируется правильно с запасом.
- \(m<0\) — объект ошибочно классифицирован (самый сильный конкурент “победил”).

В `source/main.py` строится гистограмма отступов **до** и **после** обучения (если доступен `matplotlib`):
- `students/bataev-is/task-01/margin_hist_before.png`
- `students/bataev-is/task-01/margin_hist_after.png`

---

### 3) Градиент функции потерь

Используется **квадратичная функция потерь (MSE)** на one-hot разметке:

\[
L(W) = \frac{1}{2N} \sum_{i=1}^{N} \lVert s(x_i) - y_i \rVert^2
      + \frac{\lambda}{2}\lVert W_{\text{no-bias}}\rVert^2
\]

где:
- \(s(x) = [1, x]^T W\) — линейные скоры (bias включён как первая компонента),
- \(y_i\) — one-hot вектор класса,
- \(\lambda\) — коэффициент L2-регуляризации,
- bias-строка не регуляризуется.

Градиент:

\[
\nabla_W L = \frac{1}{N} X_b^T (X_b W - Y) + \lambda \cdot \tilde{W}
\]

где \(X_b = [\mathbf{1}, X]\), а \(\tilde{W}\) — \(W\) с занулённой bias-строкой.

В `source/main.py` есть численная проверка градиента (finite differences) на подвыборке.

---

### 4) Рекуррентная оценка функционала качества

В процессе SGD ведётся рекуррентная (экспоненциальная) оценка:

\[
Q_t = (1-\alpha)Q_{t-1} + \alpha \cdot \ell_t
\]

где \(\ell_t\) — текущая (per-sample) квадратичная ошибка, \(\alpha \in (0,1)\).

---

### 5) SGD с инерцией (momentum)

Для шага SGD по одному объекту:

\[
v \leftarrow \mu v - \eta \nabla_W \ell
\]
\[
W \leftarrow W + v
\]

где \(\mu\) — momentum, \(\eta\) — learning rate.

---

### 6) L2 регуляризация

Добавляется штраф \( \frac{\lambda}{2}\lVert W_{\text{no-bias}}\rVert^2 \),
что эквивалентно добавлению в градиент члена \(\lambda W\) (кроме bias-строки).

---

### 7) Скорейший градиентный спуск

Реализован **full-batch** steepest descent с линейным поиском шага (Armijo backtracking):
- стартовый шаг `lr0`
- пока условие Армихо не выполнено — уменьшаем шаг в `beta` раз.

---

### 8) Предъявление объектов “по модулю отступа”

Вариант предъявления:
- в начале эпохи считаем \(m(x_i, y_i)\) для всех train-объектов
- сортируем по \(|m|\) по возрастанию (самые “трудные” — первыми)
- выполняем SGD в этом порядке

---

### 9) Обучение (3 режима)

В `source/main.py` автоматически запускаются:

1) **Инициализация через корреляцию (centroid-init)** + SGD(momentum)+L2  
2) **Случайная инициализация + мультистарт** (лучший по val)  
3) **Случайная инициализация + предъявление по |margin|**

---

### 10) Оценка качества

Считаются:
- accuracy на train/val/test
- confusion matrix на test для лучшей модели

---

### 11) Сравнение с эталонной реализацией

Если установлен `scikit-learn`, в конце печатается baseline:
- `sklearn.linear_model.LogisticRegression` (мультикласс)

---

### 12) Как запустить

Активировать окружение:

```bash
source /home/noru/Documents/ITMO_SUBJECTS/DL/.venv/bin/activate
```

Установить зависимости (если нужно):

```bash
pip install -U pandas tqdm matplotlib scikit-learn
```

Запуск:

```bash
python /home/noru/Documents/ITMO_SUBJECTS/fall-2025/students/bataev-is/task-01/source/main.py
```

Параметры (опционально):

```bash
python /home/noru/Documents/ITMO_SUBJECTS/fall-2025/students/bataev-is/task-01/source/main.py --epochs 120 --lr 0.03 --l2 0.001 --momentum 0.9 --multistart 30
```

---

### Реализация

- `source/model.py`: добавлен `LinearClassifier` (numpy-only) + оставлен исходный “движок” нейросети
- `source/nn_utils.py`: добавлены `linear`, `softmax`, `categorical_cross_entropy` (для совместимости; в этой лабе основное обучение идёт через MSE)
- `source/main.py`: загрузка/предобработка, эксперименты по пунктам задания, метрики/визуализации

---

### Результаты (запуск по умолчанию)

Запуск:

```bash
python /home/noru/Documents/ITMO_SUBJECTS/fall-2025/students/bataev-is/task-01/source/main.py --epochs 120 --lr 0.03 --l2 0.001 --momentum 0.9 --multistart 25
```

Получено:

- **Лучший метод по val accuracy**: `steepest_descent` (full-batch + Armijo)
  - **val acc**: 0.9667
  - **test acc**: 0.9000
- **Confusion matrix (test)**:

```
[[10  0  0]
 [ 0  8  2]
 [ 0  1  9]]
```
