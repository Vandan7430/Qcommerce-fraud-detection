import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

df = pd.read_csv("data/synthetic_qcommerce_orders.csv")

if "fraudulent" not in df.columns:
    df["fraudulent"] = np.random.randint(0, 2, size=len(df))

X = pd.get_dummies(df.drop("fraudulent", axis=1), drop_first=True)
y = df["fraudulent"]

model = RandomForestClassifier(n_estimators=20, max_depth=5)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model_rf.pkl")
joblib.dump(X.columns, "models/feature_columns.pkl")
