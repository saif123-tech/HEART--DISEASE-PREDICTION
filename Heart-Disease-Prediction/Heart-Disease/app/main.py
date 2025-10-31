from subprocess import run
from fastapi import FastAPI
from app.schema import PatientData, FeedbackData
import pickle
import numpy as np
import os
import json
from datetime import datetime
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

app = FastAPI(title="Heart Disease Predictor with Feedback")

templates = Jinja2Templates(directory="app/templates")

columns = ["age", "sex", "chest pain type", "resting bp s", "cholesterol",
           "fasting blood sugar", "resting ecg", "max heart rate",
           "exercise angina", "oldpeak", "ST slope"]


@app.get("/", response_class=HTMLResponse)
def render_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_version": model_meta.get("version"),
        "last_trained": model_meta.get("last_trained")
    })


@app.get("/api-status")
def home():
    return {"message": "🚑 Heart Disease Prediction API is active."}


# Constants
model_path = os.path.join("app", "model.pkl")
feedback_path = os.path.join("app", "feedback_data.json")
meta_path = os.path.join("app", "model_meta.json")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load model metadata
model_meta = {"version": "Unknown", "last_trained": "Not available"}
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        model_meta = json.load(f)

# Get last modified timestamp


def get_model_timestamp():
    if os.path.exists(model_path):
        ts = os.path.getmtime(model_path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    return "Unknown"


def generate_recommendations(data, prediction):
    tips = []

    # Base recommendation by prediction
    if prediction == 1:
        tips.append(
            "❗️ High risk of heart disease detected. Please consult a cardiologist.")
        tips.append("🩺 Schedule a stress test or ECG if not already done.")
        tips.append(
            "💊 Monitor blood pressure, cholesterol, and glucose regularly.")
    else:
        tips.append("✅ No current indication of heart disease.")
        tips.append("🏃 Keep up regular physical activity.")
        tips.append("🥗 Maintain a balanced diet to stay in the safe zone.")

    # Conditional health-specific tips
    if data.cholesterol > 240:
        tips.append("⚠️ Cholesterol is high — reduce fatty food intake.")
    if data.resting_blood_pressure > 140:
        tips.append("⚠️ High blood pressure — lower sodium and avoid stress.")
    if data.fasting_blood_sugar == 1:
        tips.append(
            "⚠️ Blood sugar is elevated — reduce sugar and monitor intake.")
    if data.age > 60:
        tips.append("🔁 Regular check-ups are advised for individuals over 60.")

    return tips


@app.post("/predict")
def predict(data: PatientData):
    columns = [
        "age", "sex", "chest_pain_type", "resting_blood_pressure",
        "cholesterol", "fasting_blood_sugar", "resting_ecg",
        "max_heart_rate", "exercise_angina", "oldpeak", "st_slope"
    ]

    input_data = pd.DataFrame([[
        data.age, data.sex, data.chest_pain_type, data.resting_blood_pressure,
        data.cholesterol, data.fasting_blood_sugar, data.resting_ecg,
        data.max_heart_rate, data.exercise_angina, data.oldpeak, data.st_slope
    ]], columns=columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(
        input_data)[0][1] * 100  # Probability of heart disease

    return {
        "prediction": int(prediction),
        "probability": round(probability, 2),
        "result": "Heart Disease" if prediction == 1 else "Normal"
    }


@app.post("/feedback")
def submit_feedback(feedback: FeedbackData):
    feedback_entry = feedback.dict()

    # Ensure the file exists and is valid
    if not os.path.exists(feedback_path):
        with open(feedback_path, "w") as f:
            json.dump([], f)

    # Safely load existing feedback
    with open(feedback_path, "r+") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

        data.append(feedback_entry)
        f.seek(0)
        json.dump(data, f, indent=4)

    return {"message": "✅ Feedback saved successfully!"}


@app.delete("/reset-feedback")
def reset_feedback():
    with open(feedback_path, "w") as f:
        json.dump([], f)
    return {"message": "🧹 Feedback log cleared."}


@app.get("/metrics")
def model_metrics():
    feedback_count = 0
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            try:
                feedback = json.load(f)
                feedback_count = len(feedback)
            except:
                pass

    return {
        "model_version": get_model_timestamp(),
        "feedback_count": feedback_count,
        "status": "✅ Model ready for prediction"
    }


@app.post("/feedback-ui", response_class=HTMLResponse)
def feedback_ui(
    request: Request,
    age: float = Form(...),
    sex: int = Form(...),
    chest_pain_type: int = Form(...),
    resting_blood_pressure: float = Form(...),
    cholesterol: float = Form(...),
    fasting_blood_sugar: int = Form(...),
    resting_ecg: int = Form(...),
    max_heart_rate: float = Form(...),
    exercise_angina: int = Form(...),
    oldpeak: float = Form(...),
    st_slope: int = Form(...),
    correct_label: int = Form(...)
):
    feedback_entry = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
        "correct_label": correct_label
    }

    if not os.path.exists(feedback_path):
        with open(feedback_path, "w") as f:
            json.dump([], f)

    with open(feedback_path, "r+") as f:
        try:
            data = json.load(f)
        except:
            data = []
        data.append(feedback_entry)
        f.seek(0)
        json.dump(data, f, indent=4)

    # ✅ Call retraining script
    run(["python", "retrain_model.py"], check=True)

    # 🔁 Reload the updated model into memory
    global model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "form_data": feedback_entry,
        "message": "✅ Feedback submitted and model retrained successfully!"
    })


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(request: Request,
               age: str = Form(""),
               sex: str = Form(""),
               chest_pain_type: str = Form(""),
               resting_blood_pressure: str = Form(""),
               cholesterol: str = Form(""),
               fasting_blood_sugar: str = Form(""),
               resting_ecg: str = Form(""),
               max_heart_rate: str = Form(""),
               exercise_angina: str = Form(""),
               oldpeak: str = Form(""),
               st_slope: str = Form("")):

    # Check if any field is empty
    form_data = {
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure, "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate, "exercise_angina": exercise_angina,
        "oldpeak": oldpeak, "st_slope": st_slope
    }

    # Check for empty values
    empty_fields = [field for field,
                    value in form_data.items() if not value.strip()]
    if empty_fields:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "form_data": form_data,
            "message": f"Please fill in all fields. Missing: {', '.join(empty_fields)}"
        })

    try:
        # Convert to proper types and create DataFrame
        columns = [
            "age", "sex", "chest pain type", "resting bp s",
            "cholesterol", "fasting blood sugar", "resting ecg",
            "max heart rate", "exercise angina", "oldpeak", "ST slope"
        ]

        input_data = pd.DataFrame([[
            age, sex, chest_pain_type, resting_blood_pressure,
            cholesterol, fasting_blood_sugar, resting_ecg,
            max_heart_rate, exercise_angina, oldpeak, st_slope
        ]], columns=columns)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        form_data = {
            "age": age, "sex": sex, "chest_pain_type": chest_pain_type, "resting_blood_pressure": resting_blood_pressure,
            "cholesterol": cholesterol, "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate, "exercise_angina": exercise_angina, "oldpeak": oldpeak, "st_slope": st_slope
        }

        # Convert string values to appropriate types for recommendations
        recommendations_data = {
            "age": float(age),
            "sex": int(sex),
            "chest_pain_type": int(chest_pain_type),
            "resting_blood_pressure": float(resting_blood_pressure),
            "cholesterol": float(cholesterol),
            "fasting_blood_sugar": int(fasting_blood_sugar),
            "resting_ecg": int(resting_ecg),
            "max_heart_rate": float(max_heart_rate),
            "exercise_angina": int(exercise_angina),
            "oldpeak": float(oldpeak),
            "st_slope": int(st_slope)
        }

        recommendations = generate_recommendations(
            data=type('obj', (object,), recommendations_data), prediction=prediction)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": int(prediction),
            "form_data": form_data,
            "probability": round(probability, 2),
            "message": None,
            "recommendations": recommendations,
            "model_version": model_meta.get("version"),
            "last_trained": model_meta.get("last_trained")
        })
    except ValueError as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "form_data": form_data,
            "message": f"Invalid input values. Please check your data: {str(e)}",
            "model_version": model_meta.get("version"),
            "last_trained": model_meta.get("last_trained")
        })


@app.post("/download-report")
def download_pdf(request: Request,
                 age: float = Form(...),
                 sex: int = Form(...),
                 chest_pain_type: int = Form(...),
                 resting_blood_pressure: float = Form(...),
                 cholesterol: float = Form(...),
                 fasting_blood_sugar: int = Form(...),
                 resting_ecg: int = Form(...),
                 max_heart_rate: float = Form(...),
                 exercise_angina: int = Form(...),
                 oldpeak: float = Form(...),
                 st_slope: int = Form(...)):

    # Format input
    form_data = {
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type, "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol, "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate, "exercise_angina": exercise_angina, "oldpeak": oldpeak, "st_slope": st_slope
    }

    # Predict using correct column names that match training data
    columns = [
        "age", "sex", "chest pain type", "resting bp s",
        "cholesterol", "fasting blood sugar", "resting ecg",
        "max heart rate", "exercise angina", "oldpeak", "ST slope"
    ]

    input_data = pd.DataFrame([[
        age, sex, chest_pain_type, resting_blood_pressure,
        cholesterol, fasting_blood_sugar, resting_ecg,
        max_heart_rate, exercise_angina, oldpeak, st_slope
    ]], columns=columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    recommendations = generate_recommendations(
        type("obj", (object,), form_data), prediction)

    # Create PDF using reportlab
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("🩺 Heart Disease Prediction Report", title_style))
    story.append(Spacer(1, 20))

    # Prediction Result
    result_text = "Heart Disease" if prediction == 1 else "Normal"
    result_color = colors.red if prediction == 1 else colors.green
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=14,
        textColor=result_color,
        spaceAfter=20
    )
    story.append(
        Paragraph(f"<b>Prediction Result:</b> {result_text}", result_style))
    story.append(
        Paragraph(f"<b>Confidence:</b> {round(probability, 2)}%", result_style))
    story.append(Spacer(1, 20))

    # Patient Info Table
    patient_data = [
        ['Parameter', 'Value'],
        ['Age', str(form_data['age'])],
        ['Sex', 'Male' if form_data['sex'] == 1 else 'Female'],
        ['Chest Pain Type', str(form_data['chest_pain_type'])],
        ['Resting Blood Pressure', str(form_data['resting_blood_pressure'])],
        ['Cholesterol', str(form_data['cholesterol'])],
        ['Fasting Blood Sugar', 'Yes' if form_data['fasting_blood_sugar'] == 1 else 'No'],
        ['Resting ECG', str(form_data['resting_ecg'])],
        ['Max Heart Rate', str(form_data['max_heart_rate'])],
        ['Exercise Angina', 'Yes' if form_data['exercise_angina'] == 1 else 'No'],
        ['Oldpeak', str(form_data['oldpeak'])],
        ['ST Slope', str(form_data['st_slope'])]
    ]

    patient_table = Table(patient_data, colWidths=[2*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(patient_table)
    story.append(Spacer(1, 20))

    # Recommendations
    story.append(
        Paragraph("<b>Health Recommendations:</b>", styles['Heading2']))
    story.append(Spacer(1, 10))
    for tip in recommendations:
        story.append(Paragraph(f"• {tip}", styles['Normal']))
        story.append(Spacer(1, 5))
    story.append(Spacer(1, 20))

    # Model Info
    story.append(Paragraph(
        f"<b>Model Version:</b> {model_meta.get('version')}", styles['Normal']))
    story.append(Paragraph(
        f"<b>Last Updated:</b> {model_meta.get('last_trained')}", styles['Normal']))

    # Build PDF
    doc.build(story)
    return FileResponse("report.pdf", filename="heart_disease_report.pdf", media_type="application/pdf")
