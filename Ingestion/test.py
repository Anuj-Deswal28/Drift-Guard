import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "driftguard.db"  # adjust based on where you run this from
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM requests")
print("Total rows:", cursor.fetchone())
cursor.execute("SELECT Prediction, true_lable FROM requests LIMIT 10")
print(cursor.fetchall())

conn.close()