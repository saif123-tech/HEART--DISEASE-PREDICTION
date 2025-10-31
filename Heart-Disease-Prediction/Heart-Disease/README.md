# Heart Disease Predictor

This project is a FastAPI-based web application for predicting heart disease risk using patient data. It supports feedback-driven model retraining and generates health recommendations.

## Features

- **Web Form:** Enter patient data and get predictions.
- **API Endpoints:** For prediction and feedback.
- **Model Retraining:** Incorporates user feedback to improve accuracy.
- **PDF Report Generation:** Downloadable health reports.
- **Status Endpoint:** Check API status.

## Folder Structure

```
Heart Disease/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schema.py
│   ├── model.pkl
│   ├── model_meta.json
│   ├── feedback_data.json
│   ├── report_template.html
│   ├── templates/
│   │   └── index.html
│   └── __pycache__/
│
├── retrain_model.py
├── train_model.py
├── dataset.csv
├── requirements.txt
├── report.pdf
└── README.md
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn pandas scikit-learn reportlab python-multipart
```

### 2. Run the Web App

From the project root directory (`D:\Heart Disease`):

```bash
uvicorn app.main:app --reload
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 3. Retrain the Model

After collecting feedback, run:

```bash
python retrain_model.py
```

## API Endpoints

- `GET /` — Web form for prediction interface
- `POST /predict` — Predict heart disease risk (JSON input)
- `POST /feedback` — Submit feedback for model retraining
- `GET /api-status` — Check API health status
- `GET /download-report` — Download PDF health report

## Input Features

The model expects the following patient data:

- **Age:** Patient's age in years
- **Sex:** Gender (0 = Female, 1 = Male)
- **Chest Pain Type:** Type of chest pain (0-3)
- **Resting Blood Pressure:** Resting blood pressure in mm Hg
- **Cholesterol:** Serum cholesterol in mg/dl
- **Fasting Blood Sugar:** Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)
- **Resting ECG:** Resting electrocardiographic results (0-2)
- **Max Heart Rate:** Maximum heart rate achieved
- **Exercise Angina:** Exercise induced angina (0 = No, 1 = Yes)
- **Oldpeak:** ST depression induced by exercise
- **ST Slope:** Slope of peak exercise ST segment (0-2)

## Model Training

The application uses a Random Forest Classifier with the following pipeline:
- StandardScaler for feature normalization
- RandomForestClassifier with 100 estimators and max depth of 5

## Feedback System

The application collects user feedback to improve model accuracy:
1. Users can provide corrections for predictions
2. Feedback is stored in `app/feedback_data.json`
3. Run `retrain_model.py` to incorporate feedback into the model

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- ReportLab
- Python-multipart

## Notes

- Ensure `model.pkl` and `model_meta.json` exist in the `app/` directory
- Feedback data is automatically stored in `app/feedback_data.json`
- The retraining script expects `dataset.csv` in the project root
- Run all commands from the project root directory

## Troubleshooting

### Import Errors
- Make sure `app/__init__.py` exists (can be empty)
- Run commands from the project root (`D:\Heart Disease`)
- Ensure all dependencies are installed

### Server Won't Start
- Check for duplicate route definitions
- Verify all required files exist
- Check terminal output for specific error messages

## License

MIT License
