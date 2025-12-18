import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_train_test_data():
    """Load and prepare data"""
    
    conn = sqlite3.connect('data/heart_failure.db')
    
    query = """
    SELECT 
        pd.patient_id, pd.age, pd.sex,
        cm.resting_bp, cm.cholesterol, cm.fasting_bs, cm.max_hr, cm.oldpeak,
        df.chest_pain_type, df.resting_ecg, df.exercise_angina, df.st_slope,
        hd.heart_disease
    FROM patient_demographics pd
    JOIN clinical_measurements cm ON pd.patient_id = cm.patient_id
    JOIN diagnostic_features df ON pd.patient_id = df.patient_id
    JOIN heart_disease_labels hd ON pd.patient_id = hd.patient_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Prepare
    df = df.drop('patient_id', axis=1)
    y = df['heart_disease']
    X = df.drop('heart_disease', axis=1)
    
    X = pd.get_dummies(X, drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.shape[1]
