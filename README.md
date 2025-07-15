
# 🩺 AI Heart Attack Risk Predictor

This project is a simple, interactive web app built using **Gradio**, **Pandas**, and **Scikit-learn** to predict the risk of heart disease based on user inputs. It uses a preprocessed version of the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

## 🚀 Features

- Real-time prediction of heart disease risk using a trained `RandomForestClassifier`
- Web interface built with Gradio
- Automatically trains a model if one doesn't exist
- Interactive sliders and radio buttons for user input

## 📊 Dataset

- **Source:** UCI Machine Learning Repository – [Heart Disease Data Set](http://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Target variable:** Presence of heart disease (`0 = no`, `1 = yes`)
- Missing values are filled with column median

## 🧠 Model

- Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- Accuracy is printed after training
- Model is saved to disk (`heart_attack_model.joblib`) to prevent retraining

## 🖥️ Installation

```bash
git clone https://github.com/yourusername/heart-attack-risk-predictor.git
cd heart-attack-risk-predictor
pip install -r requirements.txt
```

## 🧪 Run the App

```bash
python app.py
```

It will launch on: [http://localhost:7860](http://localhost:7860)

## 📋 Inputs

- **Age**: 1-100
- **Sex**: Male / Female
- **Chest Pain Type**: Typical Angina, Atypical Angina, etc.
- **Resting Blood Pressure**
- **Serum Cholestoral**
- **Fasting Blood Sugar**
- **ECG Results**
- **Maximum Heart Rate**
- **Exercise Induced Angina**
- **ST depression (oldpeak)**
- **Slope of ST**
- **Number of vessels colored**
- **Thalassemia**

## 📦 Requirements

Make sure to install:

```text
gradio
scikit-learn
pandas
numpy
joblib
```

Or use:

```bash
pip install -r requirements.txt
```

## ⚠️ Disclaimer

This app is intended **for educational purposes only**. It does **not** substitute professional medical advice or diagnosis.

---

Made with ❤️ using Python and Gradio
