import mlflow.pyfunc
from pathlib import Path
import pandas as pd

def load_model():
    """Load model from MLflow Registry (production way)"""
    try:
        # Load latest version from registry
        model = mlflow.pyfunc.load_model(model_uri="models:/diabetes-random-forest/latest")
        print("✅ Loaded model from MLflow Registry")
        return model
    except Exception as e:
        print(f"⚠️ Registry load failed: {e}. Trying local fallback...")
        model_path = Path("models/diabetes_model.pkl")
        if model_path.exists():
            import joblib
            return joblib.load(model_path)
        else:
            raise FileNotFoundError("Model not found. Train and register first!")

def predict_diabetes(patient_data: dict):
    model = load_model()
    
    feature_order = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    input_df = pd.DataFrame([patient_data])[feature_order]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0.5
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }
