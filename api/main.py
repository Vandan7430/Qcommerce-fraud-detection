import os
import joblib
import pandas as pd
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
    X = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True)
    y = df["fraudulent"]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")
    return model, X.columns

# Load or train
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # Load column names to match input features
    df = pd.read_csv(DATA_PATH)
    feature_columns = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True).columns
else:
    model, feature_columns = train_model()

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
    # Convert input to DataFrame
    df_input = pd.DataFrame([data.dict()])
    df_input = pd.get_dummies(df_input, drop_first=True)
    
    # Add any missing columns
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    
    prediction = model.predict(df_input)[0]
    return {"fraudulent": int(prediction)}

@app.get("/")
def home():
    return {"message": "Q-Commerce Fraud Detection API is running."}
