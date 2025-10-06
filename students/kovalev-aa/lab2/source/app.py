
from MetricAlgo import MetricClasifier
from utils import load_base,print_metrics
from sklearn.neighbors import KNeighborsClassifier
x_train,y_train = load_base()
  
model = MetricClasifier(x_train,y_train) 
model.train_k()

print(" Наша модель. Без поиска эталонов: ")
y_pred = model.predict(x_train) 
print_metrics(model.metrics(y_train, y_pred))

print("\n3. ЭТАЛОННАЯ KNN:")
knn_auto = KNeighborsClassifier()
knn_auto.fit(x_train, y_train)
y_pred_knn_auto = knn_auto.predict(x_train)
print_metrics(model.metrics(y_train, y_pred_knn_auto))


#Ищем эталоны 

model.standart_select()
y_pred = model.predict(x_train) 
print_metrics(model.metrics(y_train, y_pred))

print("После поиска эталонов: ")
