import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
from datetime import datetime

# Paths
original_dataset_path = "dataset.csv"
feedback_path = "app/feedback_data.json"
model_output_path = "app/model.pkl"
meta_output_path = "app/model_meta.json"

# 1. Load original dataset
df = pd.read_csv(original_dataset_path)
if 'class' in df.columns:
    df.rename(columns={'class': 'label'}, inplace=True)
elif 'target' in df.columns:
    df.rename(columns={'target': 'label'}, inplace=True)

# 2. Load feedback data safely
if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
    with open(feedback_path, "r") as f:
        try:
            feedback = json.load(f)
            feedback_df = pd.DataFrame(feedback)
        except json.JSONDecodeError:
            print("⚠️ Warning: feedback_data.json is corrupt or empty. Ignoring it.")
            feedback_df = pd.DataFrame()
else:
    feedback_df = pd.DataFrame()

# 3. Combine datasets
if not feedback_df.empty:
    # Map feedback column names to match original dataset
    column_mapping = {
        "chest_pain_type": "chest pain type",
        "resting_blood_pressure": "resting bp s", 
        "fasting_blood_sugar": "fasting blood sugar",
        "resting_ecg": "resting ecg",
        "max_heart_rate": "max heart rate",
        "exercise_angina": "exercise angina",
        "st_slope": "ST slope",
        "correct_label": "label"
    }
    feedback_df.rename(columns=column_mapping, inplace=True)
    combined_df = pd.concat([df, feedback_df], ignore_index=True)
else:
    combined_df = df

# 4. Train model
X = combined_df.drop("label", axis=1)
y = combined_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

pipeline.fit(X_train, y_train)

# 5. Save updated model
with open(model_output_path, "wb") as f:
    pickle.dump(pipeline, f)

# Write model metadata
meta = {
    "version": "v3.0",  # You can increment this manually or auto-update
    "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
with open(meta_output_path, "w") as f:
    json.dump(meta, f, indent=4)

print("✅ Heart disease model retrained and saved.")
print(f"📅 Model metadata updated: {meta['version']} - {meta['last_trained']}")
