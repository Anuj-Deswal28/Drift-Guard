import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "driftguard.db"  

def get_feature_data(feature_column, start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    
    query = f"SELECT {feature_column} FROM requests WHERE Time_stamp >= ? AND Time_stamp < ?"
    
    result = pd.read_sql_query(query, conn, params=(start_date, end_date))
    
    conn.close()
    return result[feature_column]

if __name__ == "__main__":
    print("sample test:-")
    data = get_feature_data("Applicant_Income", "2026-01-01", "2026-01-03")
    print(data.head())
    print(len(data))