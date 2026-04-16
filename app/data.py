from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd


TARGET = "SalePrice"


def load_data() -> pd.DataFrame:
    dataset = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    return dataset.frame.copy()


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test