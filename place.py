import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "placementdata.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

data = pd.read_csv(DATA_PATH)
data.drop(columns='StudentID', inplace=True)

values = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
label_encoders = {col: LabelEncoder() for col in values}
for col in values:
    data[col] = label_encoders[col].fit_transform(data[col])

X = data.drop(columns='PlacementStatus')
y = data['PlacementStatus']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)


os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_DIR, 'placement_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))


st.title("Placement Probability Predictor")
st.write("Please Enter the Details Below")
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
internships = st.number_input("Internships", min_value=0, max_value=10, step=1)
projects = st.number_input("Projects", min_value=0, max_value=10, step=1)
workshops = st.number_input("Workshops/Certifications", min_value=0, max_value=10, step=1)
apti_score = st.number_input("Aptitude Test Score", min_value=0, max_value=100, step=1)
soft_skills = st.number_input("Soft Skills Rating", min_value=0.0, max_value=10.0, step=0.1)
extra_activities = st.radio("Extracurricular Activities", ["No", "Yes"])
training = st.radio("Placement Training", ["No", "Yes"])
ssc_marks = st.number_input("SSC Marks", min_value=0, max_value=100, step=1)
hsc_marks = st.number_input("HSC Marks", min_value=0, max_value=100, step=1)

if st.button("Predict Placement Probability"):
    model = joblib.load(os.path.join(MODELS_DIR, 'placement_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    
    extra_activities_enc = 1 if extra_activities == "Yes" else 0
    training_enc = 1 if training == "Yes" else 0
    
    input_data = np.array([[cgpa, internships, projects, workshops, apti_score, soft_skills, extra_activities_enc, training_enc, ssc_marks, hsc_marks]])
    input_data_scaled = scaler.transform(input_data)
    
    prob = model.predict_proba(input_data_scaled)[:, 1][0] * 100
    
    st.success(f"Predicted Probability of Getting Placed: {prob:.2f}%")
