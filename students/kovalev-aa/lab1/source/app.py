import pandas as pd
import numpy as np
from datetime import datetime
from ModelsClasses import ClassifierLogisticReg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from utils import load_base,train_models,analyse_model,choose_best_model

# Загружаем данные
train_x, train_y, test_x, test_y = load_base()

# Тренировка моделей, возвращает словарь с моделями
models_dict = train_models(train_x, train_y, test_x, test_y)

# Выбор лучшей модели по accuracy 
best_name, best_model = choose_best_model(models_dict)
print(f"Max: {best_name} with accuracy {best_model.test_accuracy}")

# Анализ лучшей модели
analyse_model(best_model.model)

