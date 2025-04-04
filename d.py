pip install pycaret==3.0.0
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import load_model, predict_model

# Load the saved PyCaret model
@st.cache_resource
def load_pycaret_model():
    try:
        model = load_model('diabetes p')  # Don't include the .pkl extension
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main function for the Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.write("""
    This app predicts the likelihood of diabetes based on health metrics.
    Please enter your information below and click 'Predict' to see the results.
    """)
    
    # Load the model
    model = load_pycaret_model()
    if model is None:
        st.stop()
    
    # Create input form
    with st.form("diabetes_form"):
        st.header("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=100)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
        
        submitted = st.form_submit_button("Predict Diabetes Risk")
    
    # When the form is submitted
    if submitted:
        # Create a dataframe from the input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        # Display the input data
        st.subheader("Input Data")
        st.dataframe(input_data)
        
        # Make prediction
        try:
            prediction = predict_model(model, data=input_data)
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Get the predicted probability and class
            pred_prob = prediction['prediction_score'][0]
            pred_class = prediction['prediction_label'][0]
            
            # Display results with color coding
            if pred_class == 1:
                st.error(f"High Risk of Diabetes (Probability: {pred_prob:.2%})")
                st.write("The model predicts a high likelihood of diabetes based on the provided information.")
            else:
                st.success(f"Low Risk of Diabetes (Probability: {pred_prob:.2%})")
                st.write("The model predicts a low likelihood of diabetes based on the provided information.")
            
            # Show detailed prediction probabilities
            st.write("Detailed Prediction Probabilities:")
            st.write(f"- Probability of No Diabetes: {(1 - pred_prob):.2%}")
            st.write(f"- Probability of Diabetes: {pred_prob:.2%}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run the app
if __name__ == "__main__":
    main()
