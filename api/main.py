from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict_diabetes

app = FastAPI(title="Diabetes Prediction API")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(patient: PatientData):
    # Pydantic v2 uses model_dump() instead of dict()
    result = predict_diabetes(patient.model_dump())
    return result

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running! 🎉"}