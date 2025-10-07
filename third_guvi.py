# =============================
# Step 8: Streamlit Prediction
# =============================

import streamlit as st
import pandas as pd
import joblib

st.title("Employee Attrition Predictor")
st.markdown("Enter employee details below to check if they are likely to **stay or leave**.")


best_model = joblib.load(r"F:/GUVI/guvi_3project/best_attrition_model.pkl")
scaler = joblib.load(r"F:/GUVI/guvi_3project/scaler.pkl")          # numeric scaler
feature_names = pd.read_csv(r"F:/GUVI/guvi_3project/feature_names.csv")['Feature'].tolist()


input_data = {}
st.subheader("Employee Features")

for col in feature_names:
    if 'Gender' in col:
        input_data[col] = st.selectbox(f"{col}", options=[0, 1])  # binary encoded
    elif 'OverTime' in col:
        input_data[col] = st.selectbox(f"{col}", options=[0, 1])  # binary encoded
    else:
        input_data[col] = st.number_input(f"{col}", value=0, step=1)

input_df = pd.DataFrame([input_data])


numeric_features = scaler.feature_names_in_  # numeric columns used in scaler
input_df[numeric_features] = scaler.transform(input_df[numeric_features])


risk_score = best_model.predict_proba(input_df)[:, 1][0]

threshold = 0.3  # Custom threshold
if risk_score >= threshold:
    prediction = "Employee likely to leave ❌"
else:
    prediction = "Employee likely to stay ✅"


st.subheader("Prediction Result")
st.write(f"**Prediction:** {prediction} (Risk Score: {risk_score:.2f})")

st.subheader("Input Details")
st.dataframe(input_df.T, width=700)













