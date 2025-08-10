from fastapi import FastAPI
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

MODEL_PATH = "models/fraud_model_rf.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
DATA_PATH = "data/synthetic_qcommerce_orders.csv"

def train_model():
    """Train model if not present and save it."""
    print("⚠ Model not found — training a new one...")
    df = pd.read_csv(DATA_PATH)

    if "fraudulent" not in df.columns:
        df["fraudulent"] = np.random.randint(0, 2, size=len(df))

    X = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True)
    y = df["fraudulent"]

    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)
    print("✅ Model trained and saved!")

def load_model():
    """Load model and feature columns, train if missing."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        train_model()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns

model, feature_columns = load_model()

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df)[0]
        return {"fraudulent": bool(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Q-Commerce Fraud Detection API is running!"}
