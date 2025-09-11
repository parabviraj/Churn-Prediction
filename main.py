from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# Load pre-trained model & preprocessing objects
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

# Define expected JSON input
class ChurnInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# Preprocess JSON input
def preprocess(data: dict):
    try:
        df = pd.DataFrame([data])

        # Apply label encoders
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Scale numerical columns
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        df[num_cols] = scaler.transform(df[num_cols])

        # Reorder to match training feature order
        df = df[feature_order]

        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

# API endpoint for prediction
@app.post("/predict")
async def predict_churn(data: ChurnInput):
    try:
        df = preprocess(data.dict())
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": "Churn" if pred == 1 else "No Churn",
            "probability": round(float(prob), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Health check endpoint
@app.get("/")
async def hello_world():
    return {"message": "Churn Prediction API is running 🚀"}
