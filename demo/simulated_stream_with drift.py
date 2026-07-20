import random
from datetime import datetime, timedelta
from data.load_data import load_data
from demo.train_demo_model import train_model  
from Ingestion.ingest import log_request

def generate_random_datetime(start_date,day):
    random_second = random.randint(0, 86399)
    return start_date + timedelta(days=day, seconds=random_second)

start = datetime(2026, 1, 1)


x_train,x_test,y_train,y_test = load_data()

data = x_test.sample(n=2000,replace=True)
y_test = y_test.loc[data.index]

time_stamps =[]
for i in range(2000):
    day = i//200
    time = generate_random_datetime(start, day)
    time_stamps.append(time)
    
data["time_stamp"] = time_stamps
feature_columns = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term"
]
mod1 = train_model()

DRIFT_SHIFT = -120
DRIFT_DAYS = [7, 8, 9]

for position, (i, row) in enumerate(data.iterrows()):
    day = position // 200

    if day in DRIFT_DAYS:
        row["Credit_Score"] = row["Credit_Score"] + DRIFT_SHIFT

    row_df = row[feature_columns].to_frame().T  

    prediction = int(mod1.pred(row_df)[0])
    probability = float(mod1.pred_prob(row_df)[0][1])

    features_dict = row[feature_columns].to_dict()
    
    true_label = int(y_test.iloc[position])

    log_request(features_dict, prediction, probability, row["time_stamp"].strftime('%Y-%m-%d %H:%M:%S'), true_label)