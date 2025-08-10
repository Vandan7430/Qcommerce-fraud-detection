import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

model_path = "models/fraud_model_rf.pkl"
features_path = "models/feature_columns.pkl"

# Load model and features if available
if os.path.exists(model_path) and os.path.exists(features_path):
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    print("✅ Model and feature columns loaded successfully.")
else:
    model = None
    feature_columns = None
    print("⚠️ Warning: Model files not found. API will not be able to make predictions.")

@app.get("/")
def root():
    return {"message": "Q-Commerce Fraud Detection API is running"}

@app.post("/predict")
def predict(data: dict):
    """
    Example request:
    {
        "order_value": 250,
        "payment_mode": "CARD",
        "delivery_distance_km": 5,
        "delivery_time_minutes": 30,
        "user_order_count": 10,
        "user_avg_order_value": 200
    }
    """
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model not available. Please upload model files.")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode and align with training features
        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]
        return {"fraudulent": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

