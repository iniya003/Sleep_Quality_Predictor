import streamlit as st
import pickle
import pandas as pd

# Load trained model & encoder
model = pickle.load(open("sleep_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Advanced Sleep Quality Predictor")

# -----------------------------
# USER INPUTS
# -----------------------------
age = st.slider("Age", 10, 80, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
sleep_duration = st.slider("Sleep Duration (hours)", 3, 10, 7)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
activity = st.slider("Physical Activity Level (1-10)", 1, 10, 5)
bmi = st.slider("BMI", 15.0, 40.0, 22.0)

# -----------------------------
# DATA FOR SYMPTOMS & SOLUTIONS
# -----------------------------
sleep_info = {
    "Good": {
        "symptoms": [
            "Wake up feeling refreshed",
            "Good focus and energy levels",
            "Stable mood throughout the day"
        ],
        "solutions": [
            "Maintain your sleep routine",
            "Continue physical activity",
            "Limit screen time before bed"
        ]
    },
    "Average": {
        "symptoms": [
            "Occasional tiredness",
            "Difficulty concentrating",
            "Mild mood swings"
        ],
        "solutions": [
            "Increase sleep duration",
            "Reduce stress before bedtime",
            "Avoid caffeine at night"
        ]
    },
    "Poor": {
        "symptoms": [
            "Daytime sleepiness",
            "Frequent headaches",
            "High stress or anxiety",
            "Poor concentration"
        ],
        "solutions": [
            "Follow a strict sleep schedule",
            "Practice relaxation techniques",
            "Increase daily physical activity",
            "Consult a healthcare professional"
        ]
    }
}

# -----------------------------
# BUTTON
# -----------------------------
if st.button("Predict Sleep Quality"):
    gender_val = 1 if gender == "Male" else 0

    features = pd.DataFrame(
        [[age, gender_val, sleep_duration, stress, activity, bmi]],
        columns=["Age", "Gender", "Sleep_Duration", "Stress_Level", "Physical_Activity", "BMI"]
    )

    prediction = model.predict(features)
    result = encoder.inverse_transform(prediction)[0]

    st.subheader(f"Sleep Quality: {result}")

    if result == "Good":
        st.success(" GOOD Sleep Quality")
        st.progress(90)
    elif result == "Average":
        st.warning(" AVERAGE Sleep Quality")
        st.progress(60)
    else:
        st.error(" POOR Sleep Quality")
        st.progress(30)

    # -----------------------------
    # DISPLAY SYMPTOMS
    # -----------------------------
    st.markdown("### Symptoms")
    for s in sleep_info[result]["symptoms"]:
        st.write("•", s)

    # -----------------------------
    # DISPLAY RECOMMENDATIONS
    # -----------------------------
    st.markdown("### Recommendations")
    for r in sleep_info[result]["solutions"]:
        st.write("✔", r)
