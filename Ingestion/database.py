import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).parent


DB_PATH = BASE_DIR.parent / "data" / "driftguard.db"

conn = sqlite3.connect(DB_PATH)



cursor = conn.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY,
        Applicant_Income REAL,
        Coapplicant_Income REAL,
        Age REAL,
        Dependents REAL,
        Credit_Score REAL,
        Existing_Loans REAL,
        DTI_Ratio REAL,
        Savings REAL,
        Collateral_Value REAL,
        Loan_Amount REAL,
        Loan_Term REAL,
        Probablity REAL,
        Prediction INTEGER NOT NULL,
        Time_stamp TEXT NOT NULL,
        true_lable INTEGER
    )
''')


conn.commit()
conn.close()
