import streamlit as st
import pandas as pd
import pickle
import sklearn.compose._column_transformer as ct

# Monkey-patch to support loading old pickles that reference _RemainderColsList
class _RemainderColsList(list):
    """Stand-in for sklearn.compose._column_transformer._RemainderColsList"""
    pass
ct._RemainderColsList = _RemainderColsList

# Optionally show full tracebacks in the UI for debugging
st.set_option('client.showErrorDetails', True)

# Cache loaded assets so we don't reload on every interaction
@st.cache_resource
def load_assets():
    try:
        with open('linear_model.pkl', 'rb') as f:
            linear_model = pickle.load(f)
        with open('random_forest_model.pkl', 'rb') as f:
            random_forest_model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return linear_model, random_forest_model, preprocessor
    except Exception as e:
        st.error("Failed to load model assets. See traceback for details.")
        st.exception(e)
        st.stop()

linear_model, random_forest_model, preprocessor = load_assets()

# Predefined feature insights
feature_impact = {
    "smoker": "Smoking significantly increases medical expenses. Quitting smoking can drastically lower your costs.",
    "bmi": "High BMI indicates obesity, leading to higher medical costs. Maintaining a healthy weight can reduce expenses.",
    "age": "As age increases, medical expenses tend to rise due to higher health risks.",
    "children": "Having more dependents increases insurance costs.",
    "region": "Region has a minimal impact on charges."
}

st.title("Medical Expense Predictor")

# Sidebar inputs
st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare and preprocess data
raw_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})
processed_input = preprocessor.transform(raw_input)

if st.button("Predict Medical Expense"):
    rf_pred = random_forest_model.predict(processed_input)[0]
    st.subheader("Predicted Medical Expenses")
    st.write(f"Your estimated medical expense is: **${rf_pred:.2f}**")

    st.subheader("What Affects Your Medical Expenses?")
    for feat, impact in feature_impact.items():
        st.write(f"- **{feat.capitalize()}**: {impact}")

    st.subheader("Recommendations for Reducing Costs")
    if smoker == "yes": st.write("🚭 Quitting smoking can drastically lower your medical expenses and improve your health.")
    if bmi > 25:       st.write("🏋️ Maintaining a healthy weight through diet and exercise can reduce your healthcare costs.")
    if age > 50:       st.write("🩺 Regular check-ups and preventive care can help manage age-related health risks.")
    if children > 0:   st.write("👨‍👩‍👧‍👦 Consider family insurance plans to optimize costs for dependents.")
    st.write("🌍 Your region has minimal impact, but exploring insurers offering discounts in your area could help.")
