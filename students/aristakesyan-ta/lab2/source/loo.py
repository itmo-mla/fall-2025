import matplotlib.pyplot as plt
import numpy as np
from source.knn import KNNParzenVariableH

def loo_risk_curve(X_train, y_train, k_values):
    risks = []
    for k in k_values:
        print(f"Считаем LOO для k={k}")
        model = KNNParzenVariableH(k=k)
        model.fit(X_train, y_train)
        risk = model.loo_error()
        risks.append(risk)
        print(f"  эмпирический риск (LOO): {risk:.4f}")
    return np.array(risks)
