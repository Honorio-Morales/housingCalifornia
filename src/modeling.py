import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


def train_and_evaluate(df, features, target='MedHouseVal', test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Linear regression (multiple)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    results = {
        'linear': {'model': lr, 'r2': lr_r2, 'rmse': lr_rmse}
    }

    # Polynomial regression degrees 2 and 3
    for deg in (2, 3):
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_train_p = poly.fit_transform(X_train)
        X_test_p = poly.transform(X_test)
        lr_p = LinearRegression()
        lr_p.fit(X_train_p, y_train)
        y_pred_p = lr_p.predict(X_test_p)
        r2 = r2_score(y_test, y_pred_p)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_p))
        results[f'poly_{deg}'] = {
            'model': lr_p,
            'poly': poly,
            'r2': r2,
            'rmse': rmse
        }

    return results, X_test, y_test


def save_model(obj, path):
    joblib.dump(obj, path)


def load_model(path):
    return joblib.load(path)
