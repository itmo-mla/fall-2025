## Лабораторная работа №3 — SVM

### 1) Датасет

Использован датасет **Iris** (`source/iris.csv`) и сведён к бинарной классификации:
- классы: `versicolor` vs `virginica`
- метки: \(y \in \{-1, +1\}\), где \(y=+1\) для `virginica`
- для визуализации взяты 2 признака: `petal_length`, `petal_width`

Загрузка/EDA сделаны через **pandas**, математика SVM — на **numpy**, оптимизация — через `scipy.optimize.minimize`.

---

### 2) Решение двойственной задачи по λ (α)

Решаем soft-margin SVM в двойственной постановке:

\[
\max_{\alpha}\ \sum_{i=1}^{N}\alpha_i-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j y_i y_j K(x_i,x_j)
\]

при ограничениях:

\[
0 \le \alpha_i \le C,\qquad \sum_{i=1}^{N}\alpha_i y_i = 0
\]

Это реализовано в `source/model.py` функцией `solve_svm_dual(...)` и решается через **SLSQP** (`scipy.optimize.minimize`) с:
- bounds \(0..C\)
- линейным равенством \(\alpha^T y = 0\)
- аналитическим градиентом.

Сдвиг \(b\) восстанавливается по опорным векторам на границе \(0<\alpha_i<C\):

\[
b = \mathbb{E}_{i \in SV_{margin}}\left[y_i - \sum_j \alpha_j y_j K(x_j, x_i)\right]
\]

---

### 3) Трюк с ядром

Вместо явного скалярного произведения \(x^T x'\) используется ядро \(K(x,x')\).

Реализованы ядра:
- **linear**: \(K(x,x') = x^T x'\)
- **poly**: \(K(x,x') = (\gamma x^T x' + c_0)^{d}\)
- **rbf**: \(K(x,x') = \exp(-\gamma \|x-x'\|^2)\)

Решающая функция:

\[
f(x)=\sum_{i}\alpha_i y_i K(x_i, x) + b,\quad \hat{y}=\mathrm{sign}(f(x))
\]

---

### 4) Линейный классификатор

Для линейного ядра можно восстановить вектор весов:

\[
w = \sum_i \alpha_i y_i x_i
\]

и считать \(f(x)=w^T x + b\).

---

### 5) Визуализация

`source/main.py` строит 2D-карту решений (по сетке) и сохраняет в PNG:
- `source/svm_linear_C1.0.png`
- `source/svm_rbf_C1.0.png`

Опорные вектора выделяются отдельной обводкой.

---

### 6) Сравнение с эталонным решением

В `source/main.py` (опционально) добавлено сравнение с `sklearn.svm.SVC` для тех же параметров.

---

### Результаты (seed=42, test_ratio=0.25)

Линейное ядро (`linear`, C=1):
- train acc ≈ 0.947
- test acc ≈ 0.875
- SV ≈ 14

Ядро RBF (`rbf`, C=1, gamma=1):
- train acc ≈ 0.961
- test acc ≈ 0.917
- SV ≈ 22

Вывод: **RBF-ядро** даёт прирост качества по сравнению с линейной границей на выбранных 2 признаках.

---

### Как запустить

```bash
source /home/noru/Documents/ITMO_SUBJECTS/DL/.venv/bin/activate
pip install -U scipy pandas matplotlib scikit-learn

python /home/noru/Documents/ITMO_SUBJECTS/fall-2025/students/bataev-is/task-03/source/main.py --kernel linear --C 1.0 --gamma 1.0 --seed 42
python /home/noru/Documents/ITMO_SUBJECTS/fall-2025/students/bataev-is/task-03/source/main.py --kernel rbf --C 1.0 --gamma 1.0 --seed 42
```

### Файлы

- `source/model.py`: реализация SVM dual + kernels (numpy) + solve через scipy
- `source/main.py`: загрузка/EDA, обучение, визуализация, сравнение со sklearn
- `source/iris.csv`: датасет

