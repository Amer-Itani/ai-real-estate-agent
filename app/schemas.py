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