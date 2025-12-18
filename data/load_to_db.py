import pandas as pd
import sqlite3
import os

print("Loading data to database...")

# Read CSV
df = pd.read_csv('data/heart.csv')
print(f"Loaded {len(df)} patients")

# Create database
if os.path.exists('data/heart_failure.db'):
    os.remove('data/heart_failure.db')

conn = sqlite3.connect('data/heart_failure.db')
cursor = conn.cursor()

# Create tables
with open('data/schema.sql', 'r') as f:
    schema = f.read()
cursor.executescript(schema)

# Add patient_id
df['patient_id'] = range(1, len(df) + 1)

# Load demographics
demographics = df[['patient_id', 'Age', 'Sex']].copy()
demographics.columns = ['patient_id', 'age', 'sex']
demographics.to_sql('patient_demographics', conn, if_exists='append', index=False)

# Load clinical
clinical = df[['patient_id', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']].copy()
clinical.columns = ['patient_id', 'resting_bp', 'cholesterol', 'fasting_bs', 'max_hr', 'oldpeak']
clinical.to_sql('clinical_measurements', conn, if_exists='append', index=False)

# Load diagnostic
diagnostic = df[['patient_id', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']].copy()
diagnostic.columns = ['patient_id', 'chest_pain_type', 'resting_ecg', 'exercise_angina', 'st_slope']
diagnostic.to_sql('diagnostic_features', conn, if_exists='append', index=False)

# Load labels
labels = df[['patient_id', 'HeartDisease']].copy()
labels.columns = ['patient_id', 'heart_disease']
labels.to_sql('heart_disease_labels', conn, if_exists='append', index=False)

conn.commit()
conn.close()

print("✅ Database created!")
