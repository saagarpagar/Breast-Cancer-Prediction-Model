import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("LR_model.pkl")

st.title("🔬 Cancer Detection App")
st.write("This app predicts whether a person is likely to have cancer based on diagnostic parameters.")

# Default values that usually represent a malignant (cancer) case
mean_radius = st.number_input("Mean Radius", value=20.57)
mean_texture = st.number_input("Mean Texture", value=17.77)
mean_smoothness = st.number_input("Mean Smoothness", value=0.11840)
mean_compactness = st.number_input("Mean Compactness", value=0.27760)
mean_symmetry = st.number_input("Mean Symmetry", value=0.2419)

# Arrange inputs into model format
features = np.array([[mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_symmetry]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("⚠️ The model predicts: Cancer Detected")
    else:
        st.success("✅ The model predicts: No Cancer Detected")

st.caption("Note: These values are sample malignant case readings from the sklearn Breast Cancer dataset.")
