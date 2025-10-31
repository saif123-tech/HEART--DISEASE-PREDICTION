import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os

# Load dataset
df = pd.read_csv("dataset.csv")

# Rename target column if needed
df.rename(columns={'target': 'label'}, inplace=True)

# No encoding, just use raw columns
X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: Scaler + Classifier
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
print("\nModel Evaluation Report:\n")
print(classification_report(y_test, pipeline.predict(X_test)))

# Save model
os.makedirs("app", exist_ok=True)
with open("app/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model retrained and saved successfully.")
