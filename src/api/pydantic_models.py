from pydantic import BaseModel

class CreditApplication(BaseModel):
    # Example fields
    income: float
    age: int
    loan_amount: float
    
class PredictionResponse(BaseModel):
    approved: bool
    probability: float
