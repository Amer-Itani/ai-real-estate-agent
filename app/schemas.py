from typing import List, Optional
from pydantic import BaseModel


class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int
    YearRemodAdd: int
    TotRmsAbvGrd: int
    Neighborhood: str
    KitchenQual: str
    ExterQual: str


class PredictionResponse(BaseModel):
    predicted_price: float


class UserQuery(BaseModel):
    query: str


class FullPredictionResponse(BaseModel):
    extracted_features: dict
    missing_features: List[str]
    completeness_score: float
    predicted_price: Optional[float]
    interpretation: str
    prompt_version: str