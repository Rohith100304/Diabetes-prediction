'''import streamlit as st
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
    
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import load_model, predict_model
import base64
import io

# Load the saved PyCaret model
@st.cache_resource
def load_pycaret_model():
    try:
        model = load_model('diabetes p')  # Don't include the .pkl extension
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to create download link for files
def get_download_link(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file_path}">Download {file_label}</a>'
    return href

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
    
    # Sidebar with additional options
    with st.sidebar:
        st.header("Additional Options")
        
        # View Dataset button
        if st.button("View Dataset"):
            try:
                dataset = pd.read_csv("diabetes.csv")
                st.subheader("Diabetes Dataset")
                st.dataframe(dataset)
                st.write(f"Shape: {dataset.shape}")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        
        # Download Dataset button
        st.markdown(get_download_link("diabetes.csv", "Dataset (CSV)"), unsafe_allow_html=True)
        
        # Download Model button
        st.markdown(get_download_link("diabetes p.pkl", "Trained Model (PKL)"), unsafe_allow_html=True)
        
        st.markdown("---")
        st.write("""
        **Instructions:**
        1. Enter patient information
        2. Click 'Predict' button
        3. View results
        """)
    
    # Main content area
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
'''
import streamlit as st
import pandas as pd
import base64

# Function to create download links
def get_download_link(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file_path}">{file_label}</a>'
    return href

# Main app
def main():
    st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
    
    # ===== SIDEBAR =====
    st.sidebar.title("Options")
    
    # View Dataset button
    if st.sidebar.button("📊 View Dataset"):
        try:
            dataset = pd.read_csv("diabetes.csv")
            st.sidebar.subheader("Dataset Preview")
            st.sidebar.dataframe(dataset.head())
            st.sidebar.write(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
        except Exception as e:
            st.sidebar.error(f"Couldn't load dataset: {e}")
    
    # Download buttons
    st.sidebar.markdown("### Download Files")
    st.sidebar.markdown(get_download_link("diabetes.csv", "📥 Download Dataset"), unsafe_allow_html=True)
    st.sidebar.markdown(get_download_link("diabetes p.pkl", "📥 Download Model"), unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Enter patient data and click Predict to get results")
    
    # ===== MAIN CONTENT =====
    st.title("Diabetes Prediction App")
    
    # Your existing prediction form and logic here
    # (Keep all your existing prediction code from the previous version)
    st.write("Main prediction form would go here...")

if __name__ == "__main__":
    main()
