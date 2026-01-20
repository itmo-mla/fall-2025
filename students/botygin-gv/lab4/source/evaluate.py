from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(X_original, X_pca, y, test_size=0.2, random_state=42):
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=test_size, random_state=random_state
    )
    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state
    )

    model_original = LinearRegression()
    model_original.fit(X_train_orig, y_train)
    y_pred_orig = model_original.predict(X_test_orig)

    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    y_pred_pca = model_pca.predict(X_test_pca)

    mse_orig = mean_squared_error(y_test, y_pred_orig)
    r2_orig = r2_score(y_test, y_pred_orig)

    mse_pca = mean_squared_error(y_test, y_pred_pca)
    r2_pca = r2_score(y_test, y_pred_pca)

    return {
        "original": {"MSE": mse_orig, "R2": r2_orig},
        "pca": {"MSE": mse_pca, "R2": r2_pca}
    }