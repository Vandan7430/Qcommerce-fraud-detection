import streamlit as st
import requests

# Replace with your deployed API URL
API_URL = "https://<your-render-api-url>.onrender.com/predict"

st.set_page_config(page_title="Q-Commerce Fraud Detection", page_icon="🛒", layout="centered")

st.title("🛒 Q-Commerce Fraud Detection Dashboard")
st.write("Fill in order details to check if it's fraudulent.")

order_value = st.number_input("💰 Order Value", min_value=0.0, step=1.0)
payment_mode = st.selectbox("💳 Payment Mode", ["CARD", "CASH", "UPI"])
delivery_distance_km = st.number_input("📏 Delivery Distance (km)", min_value=0.0, step=0.1)
delivery_time_minutes = st.number_input("⏱ Delivery Time (minutes)", min_value=0, step=1)
user_order_count = st.number_input("📦 User Order Count", min_value=0, step=1)
user_avg_order_value = st.number_input("📊 User Avg Order Value", min_value=0.0, step=1.0)

if st.button("🚀 Predict Fraud"):
    payload = {
        "order_value": order_value,
        "payment_mode": payment_mode,
        "delivery_distance_km": delivery_distance_km,
        "delivery_time_minutes": delivery_time_minutes,
        "user_order_count": user_order_count,
        "user_avg_order_value": user_avg_order_value
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("fraudulent"):
                st.error("⚠️ This order is likely fraudulent!")
            else:
                st.success("✅ This order seems safe.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
