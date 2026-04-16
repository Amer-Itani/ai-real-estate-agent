from fastapi import FastAPI
import pandas as pd

from app.model import predict
from app.schemas import HouseFeatures, PredictionResponse

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.model_dump()])
    
    prediction = predict(data)[0]

    return PredictionResponse(predicted_price=prediction)