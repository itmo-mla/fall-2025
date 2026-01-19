

## Лабораторная работа №3 — SVM

### 1) Датасет

Использован датасет **Iris** (`source/iris.csv`) и сведён к бинарной классификации:

* классы: `versicolor` vs `virginica`
* метки: `y ∈ {-1, +1}`, где `y = +1` для `virginica`
* для визуализации взяты 2 признака: `petal_length`, `petal_width`

Загрузка/EDA сделаны через **pandas**, математика SVM — на **numpy**, оптимизация — через `scipy.optimize.minimize`.

---

### 2) Решение двойственной задачи по λ (α)

Решаем soft-margin SVM в двойственной постановке:

```
max_α  Σ α_i − 1/2 ΣΣ α_i α_j y_i y_j K(x_i, x_j)
```

при ограничениях:

```
0 ≤ α_i ≤ C
Σ α_i y_i = 0
```

Реализация — `solve_svm_dual(...)` в `source/model.py`, решение через **SLSQP** с:

* bounds `[0..C]`
* линейным равенством `α^T y = 0`
* аналитическим градиентом

Сдвиг `b` вычисляется по опорным векторам на границе `0 < α_i < C`:

```
b = mean_i [ y_i − Σ α_j y_j K(x_j, x_i) ]
```

---

### 3) Трюк с ядром

Вместо явного `x^T x'` используем ядро `K(x, x')`.

Реализовано:

* **linear** — `K = x^T x'`
* **poly** — `K = (γ x^T x' + c0)^d`
* **rbf** — `K = exp(−γ ||x − x'||²)`

Решающая функция:

```
f(x) = Σ α_i y_i K(x_i, x) + b
ŷ = sign(f(x))
```

---

### 4) Линейный классификатор

Для линейного ядра можно восстановить веса:

```
w = Σ α_i y_i x_i
f(x) = w^T x + b
```

---

### 5) Визуализация

`source/main.py` строит 2D-карту решений и сохраняет:

* `source/svm_linear_C1.0.png`
* `source/svm_rbf_C1.0.png`

Опорные вектора выделены на графике.

---

### 6) Сравнение с эталонным решением

Опционально можно сравнить с `sklearn.svm.SVC` для тех же параметров.

---

### Результаты (seed=42, test_ratio=0.25)

Линейное ядро (`linear`, C=1):

* train acc ≈ 0.947
* test acc ≈ 0.875
* SV ≈ 14

Ядро RBF (`rbf`, C=1, gamma=1):

* train acc ≈ 0.961
* test acc ≈ 0.917
* SV ≈ 22

**Итог:** RBF даёт лучший результат на выбранных признаках.

---

### Как запустить

```bash
source /home/noru/Documents/ITMO_SUBJECTS/DL/.venv/bin/activate
pip install -U scipy pandas matplotlib scikit-learn

python task-03/source/main.py --kernel linear --C 1.0 --gamma 1.0 --seed 42
python task-03/source/main.py --kernel rbf   --C 1.0 --gamma 1.0 --seed 42
```

---

### Файлы

* `source/model.py` — SVM dual + kernels + scipy
* `source/main.py` — загрузка/EDA, обучение, визуализация, sklearn-сравнение
* `source/iris.csv` — данные

