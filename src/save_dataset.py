from sklearn.datasets import fetch_california_housing
import pandas as pd


def save_csv(path='data/california_housing.csv'):
    b = fetch_california_housing(as_frame=True)
    df = b.frame
    df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")


if __name__ == '__main__':
    save_csv()
