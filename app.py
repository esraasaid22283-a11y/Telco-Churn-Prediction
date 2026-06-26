import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('best_churn_model.pkl')

# 2. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #10b981; color: white; height: 3em; border-radius: 10px; border: none; }
    .stButton>button:hover { background-color: #059669; }
    </style>
    """, unsafe_allow_html=True)

st.title("📡 Telco Customer Churn Predictor")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# 3. Create Inputs (Columns for organized UI)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Services")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# 4. Preprocessing User Input to Match Model Columns (Encoding)
def preprocess_input():
    features = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if online_security == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if online_backup == "Yes" else 0,
        'TechSupport_No internet service': 1 if internet_service == "No" else 0,
        'TechSupport_Yes': 1 if online_security == "Yes" else 0,
        'StreamingTV_No internet service': 1 if internet_service == "No" else 0,
        'StreamingTV_Yes': 1 if online_backup == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if internet_service == "No" else 0,
        'StreamingMovies_Yes': 1 if online_security == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }
    return pd.DataFrame([features])

# 5. Prediction
if st.button("🔮 Predict Churn Risk"):
    input_df = preprocess_input()
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"### ⚠️ High Risk: This customer is likely to CHURN!")
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.success(f"### ✅ Low Risk: This customer is likely to STAY.")
        st.write(f"Confidence Score: {1 - probability:.2%}")
