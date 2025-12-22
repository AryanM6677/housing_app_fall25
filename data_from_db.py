import sqlite3
import pandas as pd

DB_PATH = "heart.db"

def load_from_db():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM exams", con)
    con.close()
    return df
