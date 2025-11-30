import pandas as pd
import numpy as np
from datetime import datetime
from ModelsClasses import ClassifierLogisticReg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from utils import load_base,train_models,analyse_model,choose_best_model


train_x, train_y, test_x, test_y = load_base()
models_dict = train_models(train_x, train_y, test_x, test_y)

best_name, best_wrapper = choose_best_model(models_dict)
best = best_wrapper.model

print(f"Max: {best_name} with accuracy {best_wrapper.test_accuracy:.4f}")

acc_before = best_wrapper.test_accuracy 
p_before = best.test_precision
r_before = best.test_recall
f1_before = best.test_f1
analyse_model(best)
best.reset()

best.train_gd(train_x, train_y, test_x, test_y, epoches=1000, lr=0.0005, batch_count=10)

print("\n--- COMPARE GD ---")
print(f"Accuracy SGD: {acc_before:.4f} → after GD: {best.test_accuracy:.4f}")
print(f"Precision SGD: {p_before:.4f} → after GD: {best.test_precision:.4f}")
print(f"Recall SGD: {r_before:.4f} → after GD: {best.test_recall:.4f}")
print(f"F1 before SGD: {f1_before:.4f} → after GD: {best.test_f1:.4f}")

 