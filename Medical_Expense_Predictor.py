import streamlit as st
import pandas as pd
import pickle

# Load models and preprocessor
def load_assets():
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        random_forest_model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return linear_model, random_forest_model, preprocessor

linear_model, random_forest_model, preprocessor = load_assets()

# Feature Importance (pre-defined for explanation purposes)
feature_impact = {
    "smoker": "Smoking significantly increases medical expenses. Quitting smoking can drastically lower your costs.",
    "bmi": "High BMI indicates obesity, which leads to higher medical costs. Maintaining a healthy weight can reduce expenses.",
    "age": "As age increases, medical expenses tend to rise due to higher health risks.",
    "children": "Having more dependents increases insurance costs.",
    "region": "Region has a minimal impact on charges."
}

# App title
st.title("Medical Expense Predictor")

# Sidebar for user input
st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", 18, 100, 30, help="Select the patient's age")
sex = st.sidebar.selectbox("Sex", ["male", "female"], help="Select the gender of the patient")
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0, help="Enter the patient's BMI")
children = st.sidebar.slider("Children", 0, 10, 0, help="Select the number of children")
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"], help="Is the patient a smoker?")
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], help="Select the patient's region")

# Prepare input data
raw_input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocess input data to match model requirements
processed_input_data = preprocessor.transform(raw_input_data)

# Predict button
if st.button("Predict Medical Expense"):
    # Predict using Random Forest model
    rf_pred = random_forest_model.predict(processed_input_data)[0]

    # Display predictions
    st.subheader("Predicted Medical Expenses")
    st.write(f"Your estimated medical expense is: **${rf_pred:.2f}**")

    # Provide insights into feature impacts
    st.subheader("What Affects Your Medical Expenses?")
    for feature, impact in feature_impact.items():
        st.write(f"- **{feature.capitalize()}**: {impact}")

    # Personalized recommendations
    st.subheader("Recommendations for Reducing Costs")
    if smoker == "yes":
        st.write("🚭 Quitting smoking can drastically lower your medical expenses and improve your health.")
    if bmi > 25:
        st.write("🏋️ Maintaining a healthy weight through diet and exercise can reduce your healthcare costs.")
    if age > 50:
        st.write("🩺 Regular check-ups and preventive care can help manage age-related health risks.")
    if children > 0:
        st.write("👨‍👩‍👧‍👦 Consider family insurance plans to optimize costs for dependents.")
    st.write("🌍 Your region has minimal impact, but exploring insurers offering discounts in your area could help.")