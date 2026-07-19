import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "driftguard.db"  

def get_feature_data(feature_columns, start_date, end_date):
    conn = sqlite3.connect(DB_PATH)
    
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    
    columns_str = ", ".join(feature_columns) 
    
    query = f"SELECT {columns_str} FROM requests WHERE Time_stamp >= ? AND Time_stamp < ?"
    result = pd.read_sql_query(query, conn, params=(start_date, end_date))
    
    conn.close()
    
    if len(feature_columns) == 1:
        return result[feature_columns[0]]
    
    return result

if __name__ == "__main__":
    
    single = get_feature_data("Applicant_Income", "2026-01-01", "2026-01-03")
    print(type(single), len(single))
    
    
    multi = get_feature_data(["prediction", "true_lable"], "2026-01-01", "2026-01-03")
    print(type(multi))
    print(multi.head())