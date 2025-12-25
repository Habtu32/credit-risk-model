import os
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .data_processing import load_and_preprocess_data
from .predict import predict_risk
import logging

# -------------------
# Setup logging
# -------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -------------------
# FastAPI initialization
# -------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0",
    description="Predict customer credit/fraud risk using Logistic Regression model"
)

# Enable CORS for web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------
# Paths and threshold
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
THRESHOLD = 0.3

# -------------------
# Pydantic models
# -------------------
class Transaction(BaseModel):
    CustomerId: int
    TransactionId: int
    Amount: float
    TransactionStartTime: str  # ISO format

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

# -------------------
# Routes
# -------------------
@app.get("/")
def health_check():
    """API health check"""
    logging.info("Health check requested")
    return {"status": "API is running", "version": "1.0"}


@app.post("/predict")
def predict(transactions_request: TransactionsRequest):
    """
    Predict risk for a list of transactions
    """
    logging.info(f"Received {len(transactions_request.transactions)} transactions for prediction")

    # Convert request to DataFrame
    df_new = pd.DataFrame([t.dict() for t in transactions_request.transactions])

    # Predict risk
    results_df = predict_risk(df_new, MODEL_PATH, PREPROCESSOR_PATH)

    # Apply threshold
    results_df['Risk_Label'] = results_df['Predicted_Risk_Probability'].apply(
        lambda x: 1 if x >= THRESHOLD else 0
    )

    logging.info(f"Prediction completed for {len(results_df)} customers")
    return {"predictions": results_df.to_dict(orient='records')}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predict risk for a CSV file upload (batch prediction)
    """
    logging.info(f"Received CSV file: {file.filename}")
    df_new = pd.read_csv(file.file)

    # Predict risk
    results_df = predict_risk(df_new, MODEL_PATH, PREPROCESSOR_PATH)

    # Apply threshold
    results_df['Risk_Label'] = results_df['Predicted_Risk_Probability'].apply(
        lambda x: 1 if x >= THRESHOLD else 0
    )

    logging.info(f"Batch prediction completed for {len(results_df)} customers")
    return {"predictions": results_df.to_dict(orient='records')}
