import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the machine learning models
with open('diabetes_model.sav', 'rb') as model_file:
    loaded_model_diabetes = pickle.load(model_file)

with open('heart_disease_model.sav', 'rb') as model_file:
    loaded_model_heart = pickle.load(model_file)

with open('breast_cancer_model.sav', 'rb') as model_file:
    loaded_model_cancer = pickle.load(model_file)

# Load the SVC model
with open('svc.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

def make_breast_cancer_prediction(features):
    # Assuming loaded_model_cancer is your trained model
    features_array = np.array(features).reshape(1, -1)
    prediction = loaded_model_cancer.predict(features_array)
    return prediction[0]


# Load datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Define symptoms dictionary and diseases list
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56,
    'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae',
    1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox',
    11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# Function to make predictions for diabetes
def make_diabetes_prediction(input_data):
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model_diabetes.predict(input_data_reshaped)
    return prediction

# Function to make predictions for heart disease
def make_heart_disease_prediction(input_data):
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model_heart.predict(input_data_reshaped)
    return prediction

# Function to make predictions for breast cancer
def make_breast_cancer_prediction(features, scale_factor=1):
    features_array = np.array(features).reshape(1, -1) * scale_factor
    prediction = loaded_model_cancer.predict(features_array)
    return prediction[0]

# Function to predict disease based on symptoms
def get_predicted_value(patient_symptoms):
    symptoms_list = [0] * len(symptoms_dict)
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            index = symptoms_dict[symptom]
            symptoms_list[index] = 1
    return svc_model.predict([symptoms_list])[0]

# Streamlit App
st.set_page_config(layout="wide")

image_url = 'https://th.bing.com/th/id/OIP.nqlmOzQymCdXw3Mhx8BCqQHaE8?w=270&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7'
st.image('new.jpg', use_column_width=True)





tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["HOME","Disease Diagnosis", "Diabetes", "Heart Disease", "Breast Cancer", "Medications & Diets"])


with tab0:
    st.title("Welcome to MEDvision PRO+")
    st.write("Welcome to MEDvision PRO+, your comprehensive health companion.")
    
    st.subheader("Explore Health Solutions:")
    st.markdown("- **Disease Diagnosis:** Get predictions for various diseases based on symptoms.")
    st.markdown("- **Diabetes Prediction:** Predict the likelihood of diabetes based on health metrics.")
    st.markdown("- **Heart Disease Prediction:** Assess the risk of heart disease using relevant data.")
    st.markdown("- **Breast Cancer Prediction:** Predict the probability of breast cancer occurrence.")
    st.markdown("- **Medications & Diets:** Find tailored medications and diets for specific ailments.")
    st.markdown("- **Precautions:** Get precautionary measures for your health conditions ")
    
    st.subheader("How to Use:")
    st.write("Simply navigate through the tabs above to access different features.")
    st.write("Choose a feature of interest, input relevant information, and get actionable insights.")


with tab1:
    st.title("Disease Diagnosis")
    st.subheader("Predicting Diseases based on Symptoms")
    
    # Symptom selection
    selected_symptoms = st.multiselect('Select the symptoms', list(symptoms_dict.keys()))
    
    if st.button("Diagnose"):
        if selected_symptoms:
            prediction = get_predicted_value(selected_symptoms)
            disease_name = diseases_list[prediction]
            
            st.write(f"Predicted Disease: **{disease_name}**")
            
            st.write(f"Description: {description[description['Disease'] == disease_name]['Description'].values[0]}")
            st.write(f"Precautions: {precautions[precautions['Disease'] == disease_name]['Precaution_1'].values[0]}, {precautions[precautions['Disease'] == disease_name]['Precaution_2'].values[0]}, {precautions[precautions['Disease'] == disease_name]['Precaution_3'].values[0]}, {precautions[precautions['Disease'] == disease_name]['Precaution_4'].values[0]}")
        else:
            st.warning("Please select at least one symptom")

with tab2:
    st.title("Diabetes Prediction")
    st.subheader("Input data to predict Diabetes")
    
    pregnancies = st.text_input("Number of Pregnancies", "0", key="diabetes_pregnancies")
    glucose = st.text_input("Glucose Level", "0", key="diabetes_glucose")
    blood_pressure = st.text_input("Blood Pressure Value", "0", key="diabetes_blood_pressure")
    skin_thickness = st.text_input("Skin Thickness Value", "0", key="diabetes_skin_thickness")
    insulin = st.text_input("Insulin Level", "0", key="diabetes_insulin")
    bmi = st.text_input("BMI Value", "0", key="diabetes_bmi")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function Value", "0", key="diabetes_dpf")
    age = st.text_input("Age", "0", key="diabetes_age")
    
    if st.button("Predict Diabetes"):
        diabetes_prediction = make_diabetes_prediction([int(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness), float(insulin), float(bmi), float(diabetes_pedigree_function), int(age)])
        
        if diabetes_prediction[0] == 1:
            st.error("The patient is likely to have diabetes")
        else:
            st.success("The patient is unlikely to have diabetes")

with tab3:
    st.title("Heart Disease Prediction")
    st.subheader("Input data to predict Heart Disease")
    
    age = st.text_input("Age", "0", key="heart_age")
    sex = st.text_input("Sex (1 = male; 0 = female)", "0", key="heart_sex")
    cp = st.text_input("Chest Pain types", "0", key="heart_cp")
    trestbps = st.text_input("Resting Blood Pressure", "0", key="heart_trestbps")
    chol = st.text_input("Serum Cholestoral in mg/dl", "0", key="heart_chol")
    fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl", "0", key="heart_fbs")
    restecg = st.text_input("Resting Electrocardiographic results", "0", key="heart_restecg")
    thalach = st.text_input("Maximum Heart Rate achieved", "0", key="heart_thalach")
    exang = st.text_input("Exercise Induced Angina", "0", key="heart_exang")
    oldpeak = st.text_input("ST depression induced by exercise", "0", key="heart_oldpeak")
    slope = st.text_input("Slope of the peak exercise ST segment", "0", key="heart_slope")
    ca = st.text_input("Major vessels colored by flourosopy", "0", key="heart_ca")
    thal = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect", "0", key="heart_thal")
    
    if st.button("Predict Heart Disease"):
        heart_disease_prediction = make_heart_disease_prediction([int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)])
        
        if heart_disease_prediction[0] == 1:
            st.error("The patient is likely to have heart disease")
        else:
            st.success("The patient is unlikely to have heart disease")

with tab4:
    st.title("Breast Cancer Prediction")
    st.subheader("Input data to predict Breast Cancer")

    mean_radius = st.text_input("Mean Radius", "0", key="cancer_mean_radius")
    mean_texture = st.text_input("Mean Texture", "0", key="cancer_mean_texture")
    mean_perimeter = st.text_input("Mean Perimeter", "0", key="cancer_mean_perimeter")
    mean_area = st.text_input("Mean Area", "0", key="cancer_mean_area")
    mean_smoothness = st.text_input("Mean Smoothness", "0", key="cancer_mean_smoothness")
    mean_compactness = st.text_input("Mean Compactness", "0", key="cancer_mean_compactness")
    mean_concavity = st.text_input("Mean Concavity", "0", key="cancer_mean_concavity")
    mean_concave_points = st.text_input("Mean Concave Points", "0", key="cancer_mean_concave_points")
    mean_symmetry = st.text_input("Mean Symmetry", "0", key="cancer_mean_symmetry")
    mean_fractal_dimension = st.text_input("Mean Fractal Dimension", "0", key="cancer_mean_fractal_dimension")
    
    radius_error = st.text_input("Radius Error", "0", key="cancer_radius_error")
    texture_error = st.text_input("Texture Error", "0", key="cancer_texture_error")
    perimeter_error = st.text_input("Perimeter Error", "0", key="cancer_perimeter_error")
    area_error = st.text_input("Area Error", "0", key="cancer_area_error")
    smoothness_error = st.text_input("Smoothness Error", "0", key="cancer_smoothness_error")
    compactness_error = st.text_input("Compactness Error", "0", key="cancer_compactness_error")
    concavity_error = st.text_input("Concavity Error", "0", key="cancer_concavity_error")
    concave_points_error = st.text_input("Concave Points Error", "0", key="cancer_concave_points_error")
    symmetry_error = st.text_input("Symmetry Error", "0", key="cancer_symmetry_error")
    fractal_dimension_error = st.text_input("Fractal Dimension Error", "0", key="cancer_fractal_dimension_error")
    
    worst_radius = st.text_input("Worst Radius", "0", key="cancer_worst_radius")
    worst_texture = st.text_input("Worst Texture", "0", key="cancer_worst_texture")
    worst_perimeter = st.text_input("Worst Perimeter", "0", key="cancer_worst_perimeter")
    worst_area = st.text_input("Worst Area", "0", key="cancer_worst_area")
    worst_smoothness = st.text_input("Worst Smoothness", "0", key="cancer_worst_smoothness")
    worst_compactness = st.text_input("Worst Compactness", "0", key="cancer_worst_compactness")
    worst_concavity = st.text_input("Worst Concavity", "0", key="cancer_worst_concavity")
    worst_concave_points = st.text_input("Worst Concave Points", "0", key="cancer_worst_concave_points")
    worst_symmetry = st.text_input("Worst Symmetry", "0", key="cancer_worst_symmetry")
    worst_fractal_dimension = st.text_input("Worst Fractal Dimension", "0", key="cancer_worst_fractal_dimension")

    if st.button("Predict Breast Cancer"):
        features = [
            float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area), float(mean_smoothness),
            float(mean_compactness), float(mean_concavity), float(mean_concave_points), float(mean_symmetry), float(mean_fractal_dimension),
            float(radius_error), float(texture_error), float(perimeter_error), float(area_error), float(smoothness_error),
            float(compactness_error), float(concavity_error), float(concave_points_error), float(symmetry_error), float(fractal_dimension_error),
            float(worst_radius), float(worst_texture), float(worst_perimeter), float(worst_area), float(worst_smoothness),
            float(worst_compactness), float(worst_concavity), float(worst_concave_points), float(worst_symmetry), float(worst_fractal_dimension)
        ]
        breast_cancer_prediction = make_breast_cancer_prediction(features)
        
        if breast_cancer_prediction == 1:
            st.error("The patient is likely to have breast cancer")
        else:
            st.success("The patient is unlikely to have breast cancer")


with tab5:
    st.title("Medications & Diets")
    st.subheader("Find medications and diets for specific diseases")
    
    selected_disease = st.selectbox("Select a disease", description["Disease"].unique())
    
    if selected_disease:  # Check if a disease is selected
        st.write("### Medications")
        medications_list = medications[medications["Disease"] == selected_disease]["Medication"].values
        for medication in medications_list:
            st.write(f"- {medication}")
        
        st.write("### Diets")
        diets_list = diets[diets["Disease"] == selected_disease]["Diet"].values
        for diet in diets_list:
            st.write(f"- {diet}")
    else:
        st.warning("Please select a disease from the dropdown menu.")




# Footer section
st.markdown("<hr class='footer'>", unsafe_allow_html=True)

# About Us
st.markdown("<h3>About Us</h3>", unsafe_allow_html=True)
st.write(" MedVision Pro is a project by <br>SOUMYA SHUBHAM",unsafe_allow_html=True)

# More Info abou this Project
st.markdown("[More Info about this Project](www.google.com)", unsafe_allow_html=True)


# End of footer
st.markdown("<hr class='footer'>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2024 MEDvision PRO+ Web App</p>", unsafe_allow_html=True)
