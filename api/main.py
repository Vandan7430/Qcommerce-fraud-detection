import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/fraud_model_rf.pkl"
DATA_PATH = "data/synthetic_qcommerce_orders.csv"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def train_model():
    print("Training new model from CSV...")
    df = pd.read_csv(DATA_PATH)

    # If 'fraudulent' column is missing, create it with random labels
    if "fraudulent" not in df.columns:
        print("⚠ 'fraudulent' column missing. Creating random labels for demo...")
        df["fraudulent"] = np.random.randint(0, 2, size=len(df))

    # Prepare features and labels
    X = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True)
    y = df["fraudulent"]

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model and return
    joblib.dump(model, MODEL_PATH)
    print("✅ Model trained and saved.")
    return model, X.columns

# Load or train model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    if "fraudulent" not in df.columns:
        df["fraudulent"] = np.random.randint(0, 2, size=len(df))

    feature_columns = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True).columns
else:
    model, feature_columns = train_model()

# FastAPI app
app = FastAPI()

class OrderData(BaseModel):
    city: str
    order_amount: float
    items_count: int
    payment_mode: str
    time_of_day: str
    past_fraudulent_orders: int

@app.post("/predict")
def predict(data: OrderData):
    df_input = pd.DataFrame([data.dict()])
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]

    prediction = model.predict(df_input)[0]
    return {"fraudulent": int(prediction)}

@app.get("/")
def home():
    return {"message": "Q-Commerce Fraud Detection API is running."}
