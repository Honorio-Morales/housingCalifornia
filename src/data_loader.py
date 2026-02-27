from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_california_housing():
    """Carga el dataset California Housing y devuelve un DataFrame con el target.
    """
    b = fetch_california_housing(as_frame=True)
    X = b.data
    y = b.target
    df = X.copy()
    df['MedHouseVal'] = y
    return df
