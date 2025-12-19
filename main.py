from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost model trained on creditcard.csv (Time, V1–V28, Amount).",
    version="1.0.0",
)

# --- 1. Load model (JUST the model) ---
MODEL_PATH = Path("models/fraud_xgb.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model at: {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)

# --- 2. Feature list (MUST match training) ---
FEATURE_COLUMNS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Hour', 'Day', 'HourOfDay']


THRESHOLD = 0.5  
# --- 3. Request schema (one field per feature) ---
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Day: float
    HourOfDay: float
    Hour: float


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is live. Use POST /predict to score a transaction."
    }


@app.post("/predict")
def predict(tx: Transaction):
    # Convert Pydantic model -> DataFrame in correct column order
    data = tx.dict()
    row = [[data[col] for col in FEATURE_COLUMNS]]
    X = pd.DataFrame(row, columns=FEATURE_COLUMNS)

    # Model prediction
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= THRESHOLD)

    return {
        "fraud_probability": proba,
        "fraud_prediction": pred,
        "threshold_used": THRESHOLD,
    }
