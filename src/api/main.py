import pandas as pd
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API",
    version="1.0.0"
)

MODEL_URI = "models:/credit_risk_model/latest"

model = None


@app.on_event("startup")
def load_model():
    global model

    try:
        model = mlflow.pyfunc.load_model(
            MODEL_URI
        )
        print("Model loaded successfully")

    except Exception as e:
        print(
            f"Failed to load model: {e}"
        )


@app.get("/")
def root():

    return {
        "message": "Credit Risk API running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    try:

        input_df = pd.DataFrame(
            [request.model_dump()]
        )

        prediction = model.predict(
            input_df
        )[0]

        return PredictionResponse(
            risk_probability=float(prediction),
            prediction=int(
                prediction >= 0.5
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )