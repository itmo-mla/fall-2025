from knn import KNN
from data_workflow import load_and_prepare_data, scale_features, train_test_split_data
from sklearn.neighbors import KNeighborsClassifier
from metrics import Metrics
from prototype_selection import PrototypeSelector

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

selector = PrototypeSelector(X_train, y_train, k=15)

prototype_indices = selector.find_prototypes()

X_prototypes = X_train[prototype_indices]
y_prototpes = y_train[prototype_indices]

model_prototype = KNN(k=15)

model_prototype.fit(X_prototypes, y_prototpes)

y_pred_prototypes = model_prototype.predict(X_test)

print(f"Accuracy prototype: {Metrics.accuracy(y_test, y_pred_prototypes)}")

selector.plot_error_history()

selector.plot_decision_boundary()