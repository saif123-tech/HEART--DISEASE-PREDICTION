# app/schema.py

from pydantic import BaseModel


class PatientData(BaseModel):
    age: float
    sex: int
    chest_pain_type: int
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    st_slope: int


class FeedbackData(PatientData):
    correct_label: int  # 0 or 1
