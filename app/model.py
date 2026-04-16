import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict(data):
    return model.predict(data)