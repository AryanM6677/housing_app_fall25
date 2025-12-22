from data_from_db import load_from_db
from sklearn.model_selection import train_test_split

TARGET_COL = "HeartDisease"  # change if your target column has another name

def get_train_test():
    df = load_from_db()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
