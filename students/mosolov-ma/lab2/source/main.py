import numpy as np

from knn import KNN
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
from sklearn.neighbors import KNeighborsClassifier
from metrics import Metrics
from etalons_selection import PrototypeSelection

model = KNN(k=15)

etalon = KNeighborsClassifier(n_neighbors=15, metric="euclidean")

df = load_and_prepare_data()

X = df.iloc[:, :-1]

y = df['target'].to_numpy()

X = scale_features(X).to_numpy()

X_train, X_test, y_train, y_test = train_test_split_data(X, y)

etalon.fit(X_train, y_train)

model.fit(X_train, y_train)

y_pred_etalon = etalon.predict(X_test)

y_pred_model = model.predict(X_test)

print(f"Accuracy etalon: {Metrics.accuracy(y_test, y_pred_etalon)}")

print(f"Accuracy model: {Metrics.accuracy(y_test, y_pred_model)}")

selector = PrototypeSelection(k=15)
mask, history = selector.fit(X_train, y_train)
X_selected = X_train[mask]
y_selected = y_train[mask]

prototype_model = KNN(k=15)
prototype_model.fit(X_selected, y_selected)

y_pred_proto = prototype_model.predict(X_test)

print(f"Accuracy prototype: {Metrics.accuracy(y_test, y_pred_proto)}")

