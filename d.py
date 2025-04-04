import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pycaret.classification import load_model, predict_model

@st.cache_resource
def load_pycaret_model():
    try:
        model = load_model('diabetes p')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_download_link(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file_path}">{file_label}</a>'
    return href

def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
    
    with st.sidebar:
        st.title("Options")
        if st.button("View Dataset"):
            try:
                dataset = pd.read_csv("diabetes.csv")
                st.subheader("Dataset Preview")
                st.dataframe(dataset.head())
                st.write(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
            except Exception as e:
                st.error(f"Couldn't load dataset: {e}")
        
        if st.button("Download Dataset"):
            csv_link = get_download_link("diabetes.csv", "diabetes.csv")
            st.markdown(f'<a href="{csv_link}" download="diabetes.csv">Click to download dataset</a>', unsafe_allow_html=True)
        
        if st.button("Download Model"):
            pkl_link = get_download_link("diabetes p.pkl", "diabetes p.pkl")
            st.markdown(f'<a href="{pkl_link}" download="diabetes p.pkl">Click to download model</a>', unsafe_allow_html=True)
    
    st.title("Diabetes Prediction App")
    st.write("This app predicts the likelihood of diabetes based on health metrics.")
    
    model = load_pycaret_model()
    if model is None:
        st.stop()
    
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
    
    if submitted:
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
        
        st.subheader("Input Data")
        st.dataframe(input_data)
        
        try:
            prediction = predict_model(model, data=input_data)
            pred_prob = prediction['prediction_score'][0]
            pred_class = prediction['prediction_label'][0]
            
            st.subheader("Prediction Result")
            if pred_class == 1:
                st.error(f"High Risk of Diabetes (Probability: {pred_prob:.2%})")
            else:
                st.success(f"Low Risk of Diabetes (Probability: {pred_prob:.2%})")
            
            st.write("Detailed Prediction Probabilities:")
            st.write(f"- Probability of No Diabetes: {(1 - pred_prob):.2%}")
            st.write(f"- Probability of Diabetes: {pred_prob:.2%}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
