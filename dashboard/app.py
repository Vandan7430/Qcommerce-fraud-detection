import streamlit as st
import requests, os

st.set_page_config(page_title="Q-Commerce Fraud Monitor", layout="wide")
st.title("🛒 Q-Commerce Fraud Detection (API-backed)")

API_URL = st.text_input("API URL", value="http://localhost:8000/predict")

st.sidebar.header("Order Inputs")
order_value = st.sidebar.number_input("Order Value (₹)", min_value=0.0, value=500.0, step=10.0)
items_count = st.sidebar.slider("Items Count", 1, 20, 2)
hour = st.sidebar.slider("Order Hour (0-23)", 0, 23, 14)
day_of_week = st.sidebar.selectbox("Day of Week (0=Mon)", list(range(7)))
payment_mode = st.sidebar.selectbox("Payment Mode", ["COD","UPI","CARD","WALLET"])
outside_city_center = st.sidebar.selectbox("Outside City Center?", [0,1])
risky_cancel_rate = st.sidebar.selectbox("High Cancel Rate?", [0,1])
high_value_order = int(order_value > 1000)
late_night_order = int(hour in [0,1,2,3,4,22,23])

if st.button("Predict Fraud via API"):
    payload = {
        "order_value": float(order_value),
        "items_count": int(items_count),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "payment_mode": payment_mode,
        "outside_city_center": int(outside_city_center),
        "risky_cancel_rate": int(risky_cancel_rate),
        "high_value_order": int(high_value_order),
        "late_night_order": int(late_night_order)
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        if res.status_code==200:
            data = res.json()
            st.metric("Fraud Score", data.get("fraud_score"))
            st.write("Prediction:", "🚨 Fraud" if data.get("fraud_prediction")==1 else "✅ Legit")
            st.json(data)
        else:
            st.error(f"API error: {res.status_code} - {res.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
