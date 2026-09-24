
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# PAGE SETTINGS
# ------------------------------------------

st.set_page_config(
    page_title="SCI Recovery Predictor",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

MODEL_PATH = "sci_recovery_model.joblib"

model = joblib.load(MODEL_PATH)

# ------------------------------------------
# HEADER
# ------------------------------------------

st.title("🧠 SCI Recovery Predictor")

st.markdown(
    "### Clinical + MRI Machine-Learning Prototype"
)

st.warning(
    "RESEARCH PROTOTYPE — Synthetic demonstration data. "
    "Not for clinical decision-making."
)

st.divider()

# ------------------------------------------
# PATIENT INFORMATION
# ------------------------------------------

st.subheader("👤 Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

injury_level = st.selectbox(
    "Injury Level",
    ["Cervical", "Thoracic", "Lumbar"]
)

initial_ais = st.selectbox(
    "Initial AIS Grade",
    ["A", "B", "C", "D", "E"]
)

initial_motor = st.number_input(
    "Initial Motor Score",
    min_value=0,
    max_value=100,
    value=40
)

# ------------------------------------------
# MRI
# ------------------------------------------

st.subheader("🧠 MRI Findings")

edema = st.selectbox(
    "Cord Edema",
    ["Yes", "No"]
)

edema_length = st.number_input(
    "Edema Length (mm)",
    min_value=0.0,
    value=30.0
)

hemorrhage = st.selectbox(
    "MRI Hemorrhage",
    ["Yes", "No"]
)

hemorrhage_length = st.number_input(
    "Hemorrhage Length (mm)",
    min_value=0.0,
    value=0.0
)

canal_compromise = st.number_input(
    "Canal Compromise (%)",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

st.divider()

# ------------------------------------------
# PREDICT
# ------------------------------------------

if st.button(
    "🔮 PREDICT",
    type="primary",
    use_container_width=True
):

    patient = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Injury_Level": [injury_level],
        "Initial_AIS": [initial_ais],
        "Initial_Motor": [initial_motor],
        "Edema": [edema],
        "Edema_Length_mm": [edema_length],
        "Hemorrhage": [hemorrhage],
        "Hemorrhage_Length_mm": [hemorrhage_length],
        "Canal_Compromise_percent": [canal_compromise]
    })

    probability = model.predict_proba(patient)[0, 1]

    percentage = probability * 100

    # --------------------------------------
    # RESULT
    # --------------------------------------

    st.success("Prediction generated")

    st.metric(
        "Predicted probability of ≥1 AIS-grade improvement",
        f"{percentage:.1f}%"
    )

    st.progress(float(probability))

    st.info(
        "Outcome: ≥1 AIS-grade improvement at 6 months."
    )

    # --------------------------------------
    # PATIENT SUMMARY
    # --------------------------------------

    with st.expander("View Patient Summary"):

        st.dataframe(
            patient.T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    # --------------------------------------
    # DISCLAIMER
    # --------------------------------------

    st.caption(
        "Research prototype using synthetic demonstration data. "
        "This prediction must not be used for clinical decision-making."
    )
