CREATE TABLE IF NOT EXISTS patient_demographics (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clinical_measurements (
    patient_id INTEGER PRIMARY KEY,
    resting_bp INTEGER,
    cholesterol INTEGER,
    fasting_bs INTEGER,
    max_hr INTEGER,
    oldpeak REAL,
    FOREIGN KEY (patient_id) REFERENCES patient_demographics(patient_id)
);

CREATE TABLE IF NOT EXISTS diagnostic_features (
    patient_id INTEGER PRIMARY KEY,
    chest_pain_type TEXT,
    resting_ecg TEXT,
    exercise_angina TEXT,
    st_slope TEXT,
    FOREIGN KEY (patient_id) REFERENCES patient_demographics(patient_id)
);

CREATE TABLE IF NOT EXISTS heart_disease_labels (
    patient_id INTEGER PRIMARY KEY,
    heart_disease INTEGER NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patient_demographics(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_patient_id ON patient_demographics(patient_id);
CREATE INDEX IF NOT EXISTS idx_heart_disease ON heart_disease_labels(heart_disease);
