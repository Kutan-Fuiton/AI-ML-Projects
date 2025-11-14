from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

app = FastAPI()

model = load("./model/loan_model.joblib")

class LoanApplication(BaseModel):
    Gender: str | None = None
    Married: str | None = None
    Dependents: str | None = None
    Education: str | None = None
    Self_Employed: str | None = None
    ApplicantIncome: float | None = None
    CoapplicantIncome: float | None = None
    LoanAmount: float | None = None
    Loan_Amount_Term: float | None = None
    Credit_History: float | None = None
    Property_Area: str | None = None

@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    input_data = pd.DataFrame([application.dict()])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    
    return {
        "Loan_Status": result,
        "Approval_Probability": probability
    }

@app.get("/")
def home():
    return {"message": "Welcome to the Loan Approval Prediction API!"}