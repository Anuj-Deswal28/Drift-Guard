import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data():
    CSV_PATH = Path(__file__).parent / "loan_approval_data.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Loan_Approved"])
    x_data = df[["Applicant_Income","Coapplicant_Income","Age","Dependents", "Credit_Score","Existing_Loans", "DTI_Ratio","Savings","Collateral_Value","Loan_Amount", "Loan_Term" ]]
    y_data = df["Loan_Approved"]
    y_data = y_data.map({"Yes": 1, "No": 0})


    x_train, x_test, y_train, y_test =train_test_split(x_data,y_data, test_size= 0.25, random_state=32)
    train_mean = x_train.mean()
    x_train = x_train.fillna(train_mean)
    x_test = x_test.fillna(train_mean)
    
    return x_train,x_test,y_train,y_test
