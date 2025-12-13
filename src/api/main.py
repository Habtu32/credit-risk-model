from fastapi import FastAPI
from src.api.pydantic_models import CreditApplication, PredictionResponse

app = FastAPI(title="Credit Risk Model API")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(application: CreditApplication):
    # Placeholder Logic
    # In a real scenario, we would load the model and predict
    return PredictionResponse(approved=True, probability=0.95)
