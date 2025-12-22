from fastapi import FastAPI
from pydantic import BaseModel
from heart_pipeline import pipeline  # keep this

app = FastAPI(title="Heart Failure Prediction API")

class Patient(BaseModel):
    age: int
    sex: str
    resting_bp: int
    cholesterol: int
    fasting_bs: int
    max_hr: int
    oldpeak: float
    chest_pain_type: str
    resting_ecg: str
    exercise_angina: str
    st_slope: str

@app.get("/")
def read_root():
    return {"message": "Heart Failure Prediction API is running"}

@app.post("/predict")
def predict(patient: Patient):
    data = patient.dict()
    result = pipeline.predict_one(data)
    return {
        "prediction": int(result["prediction"]),
        "probability": result["probability"]
    }
