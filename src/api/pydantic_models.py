from pydantic import BaseModel

class PredictionRequest(BaseModel):

    Amount: float
    Value: float

    total_transaction_amount: float
    avg_transaction_amount: float

    transaction_count: int
    std_transaction_amount: float

    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int

class PredictionResponse(BaseModel):

    risk_probability: float
    prediction: int

