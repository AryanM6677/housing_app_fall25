# import sqlite3
# import pandas as pd
# import pickle
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# DB_PATH = os.path.join("data", "heart_failure.db")
# BEST_MODEL_PATH = os.path.join("models", "saved_models", "exp_13_SVM.pkl")  # CHANGE if file name different
import sqlite3
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Get project root: one level up from api/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "data", "heart_failure.db")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "models", "saved_models", "exp_13_SVM.pkl")  # adjust name if needed

class HeartPipeline:
    def __init__(self):
        # Load data from database
        conn = sqlite3.connect(DB_PATH)
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

        df = df.drop("patient_id", axis=1)
        self.y = df["heart_disease"]
        X = df.drop("heart_disease", axis=1)

        # One‑hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)
        self.feature_cols = X.columns.tolist()

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)

        # Fit PCA (same as training: 10 components)
        self.pca = PCA(n_components=10)
        self.pca.fit(X_scaled)

        # Load best model
        with open(BEST_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_one(self, data: dict):
        # data: raw features dict
        df = pd.DataFrame([data])

        # One‑hot encode with same columns as training
        df = pd.get_dummies(df, drop_first=True)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols]

        X_scaled = self.scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)

        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def predict_one(self, data: dict):
        X_pca = self.preprocess_one(data)
        proba = self.model.predict_proba(X_pca)[0][1]
        pred = int(proba >= 0.5)
        return {"prediction": pred, "probability": float(proba)}

pipeline = HeartPipeline()
