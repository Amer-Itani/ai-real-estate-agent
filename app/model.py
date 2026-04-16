import joblib
from pathlib import Path
import pandas as pd

MEDIAN_PRICE = 180000
MIN_PRICE = 40000
MAX_PRICE = 750000

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict(data):
    return model.predict(data)