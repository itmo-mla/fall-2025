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

## Результаты

Запуск (см. раздел **Запуск**) с параметрами:
- `--classes setosa,versicolor`
- `--max_iter 50`
- `--test_ratio 0.25`
- `--seed 42`

Коротко:
- **точность на test**: Newton–Raphson = **1.0**, IRLS = **1.0**
- **эквивалентность методов**: $\|w_{Newton} - w_{IRLS}\| = 2.219557693123839\cdot 10^{-14}$

Коэффициенты (bias + 4 признака):

| Метод | $w_0$ (bias) | $w_1$ | $w_2$ | $w_3$ | $w_4$ |
|---|---:|---:|---:|---:|---:|
| Newton–Raphson | 1.20122440 | 2.18956153 | -3.29947092 | 6.44224737 | 5.70817982 |
| IRLS | 1.20122440 | 2.18956153 | -3.29947092 | 6.44224737 | 5.70817982 |

<details>
<summary>Фактический вывод скрипта (stdout)</summary>

```text
Dataset: /home/noru/Documents/fall-2025/students/bataev-is/task-05/source/iris.csv
Binary classes: ['setosa', 'versicolor'] -> y=1 is versicolor

Coefficients (bias + 4 features):
w_newton: [ 1.2012244   2.18956153 -3.29947092  6.44224737  5.70817982]
w_irls  : [ 1.2012244   2.18956153 -3.29947092  6.44224737  5.70817982]
||w_newton - w_irls||: 2.219557693123839e-14

Test accuracy:
Newton-Raphson: 1.0
IRLS         : 1.0
```

</details>

---

## Выводы

- **Newton–Raphson и IRLS дают одинаковое решение** (норма разности весов близка к машинному нулю), что подтверждает их теоретическую эквивалентность для логистической регрессии.
- **Качество на Iris (setosa vs versicolor)** при выбранном разбиении и стандартизации получилось **идеальным (accuracy = 1.0)**, т.к. классы линейно хорошо разделимы в пространстве признаков.
- **Стандартизация важна**: улучшает численную устойчивость и делает масштаб признаков сопоставимым, что влияет на устойчивость решения системы на каждом шаге.
- **Малый L2 ($\lambda \approx 10^{-6}$)** использован как техническая регуляризация для устойчивого решения линейных систем (bias не регуляризуется) и практически не меняет MLE на этом датасете.

---

## Структура проекта

- `source/main.py` — реализация логистической регрессии (Newton/IRLS), предобработка, запуск эксперимента, опциональные графики/сравнение со sklearn.
- `source/iris.csv` — датасет.
