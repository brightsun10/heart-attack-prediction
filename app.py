import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. DATA LOADING AND MODEL TRAINING ---

# Function to load data and train the model
def train_model():
    # Load the dataset from the UCI repository
    # This is a well-known dataset for heart disease prediction
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    # Read the data, treating '?' as missing values
    df = pd.read_csv(url, header=None, names=column_names, na_values='?')

    # --- Data Preprocessing ---
    # For simplicity, we'll fill missing values with the median of their column
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # The 'target' column is our label. 0 = no disease, 1-4 = varying degrees of disease.
    # We will simplify this to a binary classification: 0 = no disease, 1 = disease present.
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Define features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    # We use a RandomForestClassifier, which is robust and performs well on this type of data.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with Accuracy: {accuracy:.2f}")

    return model

# Check if a trained model file exists, otherwise train a new one
model_filename = "heart_attack_model.joblib"
if not os.path.exists(model_filename):
    print("No pre-trained model found. Training a new one...")
    model = train_model()
    # Save the trained model to a file for future use
    joblib.dump(model, model_filename)
else:
    print("Loading pre-trained model.")
    model = joblib.load(model_filename)


# --- 2. PREDICTION FUNCTION ---

def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Makes a prediction based on user input from the Gradio interface.
    """
    # Create a pandas DataFrame from the user inputs
    # The order of columns must match the order used during training
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Make a prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Return a user-friendly result
    if prediction == 1:
        result = "High Risk of Heart Disease"
        probability = prediction_proba[1]
    else:
        result = "Low Risk of Heart Disease"
        probability = prediction_proba[0]

    return f"{result} (Confidence: {probability:.2f})"


# --- 3. GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft(), title="Heart Attack Risk Predictor") as demo:
    gr.Markdown("# 🩺 AI Heart Attack Risk Predictor")
    gr.Markdown(
        "Enter the patient's medical information below to get a prediction on their risk of heart disease. "
        "This tool is for educational purposes only and is not a substitute for professional medical advice."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Demographics")
            age = gr.Slider(label="Age", minimum=1, maximum=100, value=50)
            sex = gr.Radio( choices=[1, 0], value=1, label="Sex (1 = Male, 0 = Female)")

            
            gr.Markdown("### Symptoms & Vitals")
            cp = gr.Radio(label="Chest Pain Type (cp) | (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal Pain, 4 = Asymptomatic)", choices=[1, 2, 3, 4,], value=1)
            trestbps = gr.Slider(label="Resting Blood Pressure (trestbps) in mm Hg", minimum=80, maximum=220, value=120)
            chol = gr.Slider(label="Serum Cholestoral (chol) in mg/dl", minimum=100, maximum=600, value=200)
            fbs = gr.Radio(label="Fasting Blood Sugar > 120 mg/dl (fbs) | (1 = true, 0 = False)", choices=[1, 0], value=0)

        with gr.Column():
            gr.Markdown("### Test Results")
            restecg = gr.Radio(label="Resting Electrocardiographic Results (restecg) | (0 = Normal, 1 = ST-T wave abnormality, 2 = Probable or definite left ventricular hypertrophy)", choices=[0, 1, 2], value=0)
            thalach = gr.Slider(label="Maximum Heart Rate Achieved (thalach)", minimum=60, maximum=220, value=150)
            exang = gr.Radio(label="Exercise Induced Angina (exang) | (1 = Yes, 0 = No)", choices=[1, 0], value=0)
            oldpeak = gr.Slider(label="ST depression induced by exercise relative to rest (oldpeak)", minimum=0.0, maximum=7.0, step=0.1, value=1.0)
            slope = gr.Radio(label="Slope of the peak exercise ST segment (slope) | (1 = Upsloping, 2 = Flat, 3 = Downsloping)", choices=[1, 2, 3], value=1)
            ca = gr.Slider(label="Number of major vessels colored by flourosopy (ca)", minimum=0, maximum=3, step=1, value=0)
            thal = gr.Radio(label="Thalassemia (thal) | (3 = Normal, 6 = Fixed defect, 7 = Reversable defect)", choices=[3, 6, 7], value=3)

    predict_button = gr.Button("Predict Risk", variant="primary")
    
    output_text = gr.Textbox(label="Prediction Result", interactive=False)

    predict_button.click(
        fn=predict_heart_attack,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=output_text
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(debug=True)
