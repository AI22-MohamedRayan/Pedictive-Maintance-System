import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("failure_model.pkl")
cols = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Machine Failure Prediction", layout="centered")

st.title("🔧 Machine Failure Prediction")
st.write("Enter the machine details below.")

# Numerical Inputs
operational_hours = st.number_input("Operational Hours", min_value=0.0, value=1000.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=45.0)
vibration = st.number_input("Vibration (mm/s)", min_value=0.0, value=5.0)
sound = st.number_input("Sound (dB)", min_value=0.0, value=70.0)
oil = st.number_input("Oil Level (%)", min_value=0.0, max_value=100.0, value=60.0)
coolant = st.number_input("Coolant Level (%)", min_value=0.0, max_value=100.0, value=65.0)
power = st.number_input("Power Consumption (kW)", min_value=0.0, value=12.0)
last_maint = st.number_input("Last Maintenance Days Ago", min_value=0, value=15)
maint_history = st.number_input("Maintenance History Count", min_value=0, value=3)
failure_history = st.number_input("Failure History Count", min_value=0, value=1)
ai_supervision = st.selectbox("AI Supervision", [0, 1])
error_codes = st.number_input("Error Codes in Last 30 Days", min_value=0, value=2)
ai_override = st.number_input("AI Override Events", min_value=0, value=0)
machine_age = st.number_input("Machine Age (years)", min_value=0, value=5)

# Machine type dropdown
machine_types = [
    "3D_Printer",
    "AGV",
    "Automated_Screwdriver",
    "Boiler",
    "CMM",
    "CNC_Lathe",
    "CNC_Mill",
    "Carton_Former",
    "Compressor",
    "Conveyor_Belt",
    "Crane",
    "Dryer",
    "Forklift_Electric",
    "Furnace",
    "Grinder",
    "Heat_Exchanger",
    "Hydraulic_Press",
    "Industrial_Chiller",
    "Injection_Molder",
    "Labeler",
    "Laser_Cutter",
    "Mixer",
    "Palletizer",
    "Pick_and_Place",
    "Press_Brake",
    "Pump",
    "Robot_Arm",
    "Shrink_Wrapper",
    "Shuttle_System",
    "Vacuum_Packer",
    "Valve_Controller",
    "Vision_System",
    "XRay_Inspector"
]

machine_type = st.selectbox("Machine Type", machine_types)

# Initialize all columns to 0
input_data = {c: 0 for c in cols}

# Fill numeric columns
input_data["Operational_Hours"] = operational_hours
input_data["Temperature_C"] = temperature
input_data["Vibration_mms"] = vibration
input_data["Sound_dB"] = sound
input_data["Oil_Level_pct"] = oil
input_data["Coolant_Level_pct"] = coolant
input_data["Power_Consumption_kW"] = power
input_data["Last_Maintenance_Days_Ago"] = last_maint
input_data["Maintenance_History_Count"] = maint_history
input_data["Failure_History_Count"] = failure_history
input_data["AI_Supervision"] = ai_supervision
input_data["Error_Codes_Last_30_Days"] = error_codes
input_data["AI_Override_Events"] = ai_override
input_data["Machine_Age"] = machine_age

# Set selected machine type column to 1
# 3D_Printer is the default class and has no separate column
machine_col = f"Machine_Type_{machine_type}"

if machine_col in input_data:
    input_data[machine_col] = 1


if "Failure_Within_7_Days" in input_data:
    del input_data["Failure_Within_7_Days"]

input_df = pd.DataFrame([input_data])


model_input_cols = [c for c in cols if c != "Failure_Within_7_Days"]
input_df = input_df[model_input_cols]


if st.button("Predict Failure"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ Machine is likely to fail within 7 days")
    else:
        st.success("✅ Machine is not likely to fail within 7 days")

    st.write(f"Failure Probability: {probability:.2%}")

    with st.expander("Input Used for Prediction"):
        st.dataframe(input_df)