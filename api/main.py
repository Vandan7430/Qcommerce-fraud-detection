from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, os, pandas as pd

app = FastAPI(title="Q-Commerce Fraud Detection API")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model = joblib.load(os.path.join(BASE, "models", "fraud_model_rf.pkl"))
encoder = joblib.load(os.path.join(BASE, "models", "payment_mode_encoder.joblib"))
with open(os.path.join(BASE, "models", "feature_names.json")) as f:
    feature_names = json.load(f)

class Order(BaseModel):
    order_value: float
    items_count: int
    hour: int
    day_of_week: int
    payment_mode: str
    outside_city_center: int
    risky_cancel_rate: int
    high_value_order: int
    late_night_order: int

@app.post("/predict")
def predict(order: Order):
    data = pd.DataFrame([order.dict()])
    pm_encoded = encoder.transform(data[["payment_mode"]])
    pm_cols = encoder.get_feature_names_out(['payment_mode']).tolist()
    pm_df = pd.DataFrame(pm_encoded, columns=pm_cols)
    X = pd.concat([data.drop(columns=['payment_mode']).reset_index(drop=True), pm_df.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=feature_names, fill_value=0)
    prob = float(model.predict_proba(X)[:,1][0])
    pred = int(model.predict(X)[0])
    return { "fraud_prediction": pred, "fraud_score": round(prob,4) }
