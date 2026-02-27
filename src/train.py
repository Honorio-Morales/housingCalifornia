from data_loader import load_california_housing
from modeling import train_and_evaluate, save_model
import os


FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Latitude', 'Longitude']


def main():
    df = load_california_housing()
    results, X_test, y_test = train_and_evaluate(df, FEATURES)

    os.makedirs('models', exist_ok=True)

    # Guardar modelos
    save_model(results['linear']['model'], 'models/model_linear.joblib')
    save_model(results['poly_2']['model'], 'models/model_poly_2.joblib')
    save_model(results['poly_2']['poly'], 'models/poly_2_transform.joblib')
    save_model(results['poly_3']['model'], 'models/model_poly_3.joblib')
    save_model(results['poly_3']['poly'], 'models/poly_3_transform.joblib')

    # Imprimir métricas
    for name, info in results.items():
        print(f"{name}: R2={info['r2']:.4f}, RMSE={info['rmse']:.4f}")


if __name__ == '__main__':
    main()
