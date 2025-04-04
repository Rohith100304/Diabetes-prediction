import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import base64

# Load the saved model and dataset
@st.cache_resource
def load_heart_model():
    return load_model('heart')

@st.cache_data
def load_dataset():
    return pd.read_csv('heart.csv')

model = load_heart_model()
data = load_dataset()

# File download function
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def pkl_download(model, filename):
    with open(f'{model}.pkl', 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'<a href="data:file/pkl;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Create a function to get user input in main area
def get_user_input():
    st.header("Patient Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=70, max_value=220, value=150)
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0)
    
    with col2:
        # Categorical inputs
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        restecg = st.selectbox('Resting Electrocardiographic Results', 
                              ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                            ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', 
                           ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == 'Male' else 0
    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp = cp_mapping[cp]
    fbs = 1 if fbs == 'Yes' else 0
    restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == 'Yes' else 0
    slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 0}
    slope = slope_mapping[slope]
    thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
    thal = thal_mapping[thal]
    
    # Store a dictionary into a dataframe
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    # Title
    st.title('Heart Disease Prediction App')
    st.write("""
    This app predicts the likelihood of having heart disease based on patient information.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    # Sidebar options
    st.sidebar.title("Options")
    
    if st.sidebar.checkbox("View Dataset"):
        st.subheader("Heart Disease Dataset")
        st.write(data)
        st.markdown(filedownload(data, 'heart.csv'), unsafe_allow_html=True)
    
    st.sidebar.markdown("### Download Model")
    st.sidebar.markdown(pkl_download('heart', 'heart_model.pkl'), unsafe_allow_html=True)
    
    # Get user input in main area
    user_input = get_user_input()
    
    # Display user input
    st.subheader('Patient Input Summary')
    st.write(user_input)
    
    # Prediction button
    if st.button('Predict Heart Disease Risk'):
        # Make prediction
        prediction = predict_model(model, data=user_input)
        
        # Display prediction
        st.subheader('Prediction Result')
        prediction_value = prediction['prediction_label'][0]
        prediction_score = prediction['prediction_score'][0]
        
        if prediction_value == 1:
            st.error(f'**High risk of heart disease** (Probability: {prediction_score:.2%})')
            st.warning('Please consult with a healthcare professional for further evaluation.')
        else:
            st.success(f'**Low risk of heart disease** (Probability: {1 - prediction_score:.2%})')
            st.info('Maintain a healthy lifestyle for continued heart health.')
            
        # Show prediction details expander
        with st.expander("Show detailed prediction metrics"):
            st.write(prediction)

if __name__ == '__main__':
    main()
'''import streamlit as st
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
                st.write("Dataset Preview")
                st.dataframe(dataset)
                st.write(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
            except Exception as e:
                st.error(f"Couldn't load dataset: {e}")
        
        if st.button("Download Dataset"):
            csv_link = get_download_link("diabetes.csv", "diabetes.csv")
            st.markdown(f'{csv_link}', unsafe_allow_html=True)
        
        if st.button("Download Model"):
            pkl_link = get_download_link("diabetes p.pkl", "diabetes p.pkl")
            st.markdown(f'{pkl_link}', unsafe_allow_html=True)
    
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
'''
