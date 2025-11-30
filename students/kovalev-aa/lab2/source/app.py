from MetricAlgo import MetricClasifier
from utils import load_base, print_metrics
from sklearn.neighbors import KNeighborsClassifier

x_train, y_train = load_base()

# Наша модель
model = MetricClasifier(x_train, y_train)
model.train_k()

print("Наша модель. Без поиска эталонов:")
y_pred = model.predict(x_train)
print_metrics(model.metrics(y_train, y_pred))

 
print("\nЭТАЛОННАЯ KNN (weights='distance'):")
knn_auto = KNeighborsClassifier(weights='distance')
knn_auto.fit(x_train, y_train)
y_pred_knn_auto = knn_auto.predict(x_train)
print_metrics(model.metrics(y_train, y_pred_knn_auto))

# Ищем эталоны
model.standart_select()
y_pred = model.predict(x_train)
print("\nПосле поиска эталонов (наша модель):")
print_metrics(model.metrics(y_train, y_pred))
