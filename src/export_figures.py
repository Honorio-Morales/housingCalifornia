import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Use seaborn theme (safer than plt.style.use with missing style name)
sns.set_theme(style='whitegrid')

OUTDIR = 'reports/figures'
os.makedirs(OUTDIR, exist_ok=True)

FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Latitude', 'Longitude']
TARGET = 'MedHouseVal'


def save_heatmap(df):
    corr = df[FEATURES + [TARGET]].corr()
    fpath = os.path.join(OUTDIR, 'heatmap_correlation.png')
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap - Correlaciones (features + target)')
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    print('Saved', fpath)


def save_scatter_top_features(df, top_n=3):
    corr = df[FEATURES + [TARGET]].corr()
    corr_target = corr[TARGET].sort_values(ascending=False)
    top_features = corr_target.index[1:1+top_n].tolist()
    fpath = os.path.join(OUTDIR, 'scatter_top_features.png')
    plt.figure(figsize=(5*top_n,4))
    for i, f in enumerate(top_features, 1):
        plt.subplot(1, top_n, i)
        sns.scatterplot(x=df[f], y=df[TARGET], alpha=0.4)
        plt.xlabel(f)
        plt.ylabel(TARGET)
        plt.title(f'{f} vs {TARGET}')
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    print('Saved', fpath)


def plot_pred_resid_save(y_true, y_pred, name):
    fpath = os.path.join(OUTDIR, f'pred_resid_{name}.png')
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    plt.title(f'Real vs Predicho - {name}')

    plt.subplot(1,2,2)
    resid = y_true - y_pred
    sns.scatterplot(x=y_pred, y=resid, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicho')
    plt.ylabel('Residual')
    plt.title(f'Residuals - {name}')

    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    print('Saved', fpath)


def main():
    # Cargar dataset
    df = pd.read_csv('data/california_housing.csv')

    save_heatmap(df)
    save_scatter_top_features(df, top_n=3)

    # Train/test split
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cargar modelos
    models_found = {}
    try:
        models_found['linear'] = joblib.load('models/model_linear.joblib')
    except Exception as e:
        print('Linear model not found:', e)
    try:
        models_found['poly2'] = joblib.load('models/model_poly_2.joblib')
        models_found['poly2_poly'] = joblib.load('models/poly_2_transform.joblib')
    except Exception as e:
        print('Poly2 model not found:', e)
    try:
        models_found['poly3'] = joblib.load('models/model_poly_3.joblib')
        models_found['poly3_poly'] = joblib.load('models/poly_3_transform.joblib')
    except Exception:
        # optional
        pass

    # Linear preds
    if 'linear' in models_found:
        y_pred_lr = models_found['linear'].predict(X_test)
        r2 = r2_score(y_test, y_pred_lr)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        print(f'Linear: R2={r2:.4f}, RMSE={rmse:.4f}')
        plot_pred_resid_save(y_test, y_pred_lr, 'linear')

    # Poly2 preds
    if 'poly2' in models_found and 'poly2_poly' in models_found:
        X_test_p2 = models_found['poly2_poly'].transform(X_test)
        y_pred_p2 = models_found['poly2'].predict(X_test_p2)
        r2 = r2_score(y_test, y_pred_p2)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_p2))
        print(f'Poly2: R2={r2:.4f}, RMSE={rmse:.4f}')
        plot_pred_resid_save(y_test, y_pred_p2, 'poly2')

    # Poly3 preds (if present)
    if 'poly3' in models_found and 'poly3_poly' in models_found:
        X_test_p3 = models_found['poly3_poly'].transform(X_test)
        y_pred_p3 = models_found['poly3'].predict(X_test_p3)
        r2 = r2_score(y_test, y_pred_p3)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_p3))
        print(f'Poly3: R2={r2:.4f}, RMSE={rmse:.4f}')
        plot_pred_resid_save(y_test, y_pred_p3, 'poly3')

    print('All figures saved in', OUTDIR)


if __name__ == '__main__':
    main()
