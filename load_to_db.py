import sqlite3
import pandas as pd

CSV_PATH = "data/heart.csv"
DB_PATH = "heart.db"
TABLE_NAME = "exams"

def main():
    print("Loading data to database...")

    df = pd.read_csv(CSV_PATH)
    print("CSV rows:", len(df))

    con = sqlite3.connect(DB_PATH)

    # write table "exams"
    df.to_sql(TABLE_NAME, con, if_exists="replace", index=False)

    con.commit()
    con.close()

    print(f"Loaded {len(df)} rows into table '{TABLE_NAME}'")
    print("✅ Database created!")

if __name__ == "__main__":
    main()
