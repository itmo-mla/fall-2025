model = SGD(100000, 1e-5, 0.001, 5e-2, 0.9, 0.01, 'rand', 'corr', 10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_margins(model,X_test, y_test)