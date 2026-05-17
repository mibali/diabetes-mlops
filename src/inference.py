
import joblib
from pathlib import Path
import pandas as pd

def load_model():
    model_path = Path("models/diabetes_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Model not found. Please run training first!")

def predict_diabetes(patient_data: dict):
    model = load_model()
    
    # Make sure column order matches training data
    feature_order = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    input_df = pd.DataFrame([patient_data])[feature_order]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }
