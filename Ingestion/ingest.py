import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "driftguard.db"

def log_request(features, prediction, probability, timestamp, true_lable=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO requests (
            Applicant_Income, Coapplicant_Income, Age, Dependents,
            Credit_Score, Existing_Loans, DTI_Ratio, Savings,
            Collateral_Value, Loan_Amount, Loan_Term,
            Probablity, Prediction, Time_stamp, true_lable
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
    features["Applicant_Income"], features["Coapplicant_Income"], features["Age"],
    features["Dependents"], features["Credit_Score"], features["Existing_Loans"],
    features["DTI_Ratio"], features["Savings"], features["Collateral_Value"],
    features["Loan_Amount"], features["Loan_Term"],
    probability, prediction, timestamp, true_lable
)
    )
    conn.commit()
    conn.close()
    
if __name__ == "__main__":

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM requests LIMIT 3")
    print("Insertion successful")
    print(cursor.fetchall())
    conn.close()