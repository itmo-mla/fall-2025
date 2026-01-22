## Лабораторная работа №5 — Логистическая регрессия (Newton–Raphson / IRLS)

Реализация **бинарной** логистической регрессии на `numpy` с двумя эквивалентными оптимизаторами:
- **Newton–Raphson**
- **IRLS** (Iteratively Reweighted Least Squares)

Также есть предобработка (stratified split + стандартизация) на `numpy`/`pandas` и опциональное сравнение со `sklearn`.

---

## Датасет

Используется **Iris**: [`source/iris.csv`](source/iris.csv). Задача сведена к бинарной классификации:
- **классы**: `setosa` vs `versicolor`
- **целевая переменная**: $y=1$ для `versicolor`, $y=0$ для `setosa`
- **признаки**: 4 числовых (стандартные признаки Iris)
- **предобработка**:
  - stratified train/test split
  - стандартизация признаков по train

---

## Модель и функция потерь

Модель:

$$
p(y=1 \mid x) = \sigma(w^\top x_b), \qquad \sigma(z)=\frac{1}{1+e^{-z}}
$$

где $x_b = [1, x]^\top$ — вектор признаков с bias.

Оптимизируем **средний отрицательный лог-правдоподобие** (NLL):

$$
\mathcal{L}(w) = -\frac{1}{N}\sum_{i=1}^N \Big(y_i \log p_i + (1-y_i)\log(1-p_i)\Big)
$$

В реализации добавлен очень маленький L2 ($\lambda \approx 10^{-6}$) **только для численной устойчивости** (bias не регуляризуется).

---

## Newton–Raphson

Градиент:

$$
\nabla \mathcal{L}(w) = \frac{1}{N} X_b^\top (p - y) + \lambda \tilde{w}
$$

Гессиан:

$$
H = \frac{1}{N} X_b^\top R X_b + \lambda I, \qquad
R = \mathrm{diag}(p_i(1-p_i))
$$

Шаг Ньютона:

$$
w \leftarrow w - H^{-1}\nabla \mathcal{L}(w)
$$

---

## IRLS и эквивалентность шагу Ньютона

IRLS строит “рабочий отклик”:

$$
z = X_b w,\quad p=\sigma(z),\quad R_i=p_i(1-p_i),\quad
z^{(work)} = z + \frac{y - p}{R}
$$

и решает взвешенную МНК-задачу:

$$
w \leftarrow \arg\min_w \left\| R^{1/2}(z^{(work)} - X_b w)\right\|^2
$$

что приводит к системе:

$$
(X_b^\top R X_b)\, w = X_b^\top R z^{(work)}
$$

Для логистической регрессии это **эквивалентно шагу Newton–Raphson**.

---

## Запуск

Из директории `task-05`:

```bash
python source/main.py --classes setosa,versicolor --max_iter 50 --test_ratio 0.25 --seed 42
```

Опционально:
- для графика сходимости нужен `matplotlib` (файл `nll_convergence.png`)
- для сравнения с эталоном нужен `scikit-learn` (`sklearn.linear_model.LogisticRegression`)

---

## Ожидаемые результаты

На `setosa` vs `versicolor` обычно получается:
- **эквивалентность решений**: $\|w_{Newton} - w_{IRLS}\| \approx 2.22\cdot 10^{-14}$
- **точность на test**: 1.0 (для обоих методов)

---

## Структура проекта

- `source/main.py` — реализация логистической регрессии (Newton/IRLS), предобработка, запуск эксперимента, опциональные графики/сравнение со sklearn.
- `source/iris.csv` — датасет.
