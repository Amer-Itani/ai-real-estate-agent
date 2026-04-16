from fastapi import FastAPI
import pandas as pd

from app.model import predict
from app.schemas import HouseFeatures, PredictionResponse
from app.llm_stage1 import extract_features_from_query
from app.llm_stage2 import interpret_prediction
from app.schemas import FullPredictionResponse, UserQuery

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.model_dump()])
    
    prediction = predict(data)[0]

    return PredictionResponse(predicted_price=prediction)

@app.post("/analyze-query", response_model=FullPredictionResponse)
def analyze_query(payload: UserQuery):
    stage1 = extract_features_from_query(payload.query)

    if stage1["missing_features"]:
        return FullPredictionResponse(
            extracted_features=stage1["extracted_features"],
            missing_features=stage1["missing_features"],
            completeness_score=stage1["completeness_score"],
            predicted_price=None,
            interpretation="Some features are missing. Please refine your input.",
            prompt_version=stage1["prompt_version"],
        )

    df = pd.DataFrame([stage1["extracted_features"]])
    prediction = float(predict(df)[0])

    explanation = interpret_prediction(stage1["extracted_features"], prediction)

    return FullPredictionResponse(
        extracted_features=stage1["extracted_features"],
        missing_features=stage1["missing_features"],
        completeness_score=stage1["completeness_score"],
        predicted_price=prediction,
        interpretation=explanation,
        prompt_version=stage1["prompt_version"],
    )