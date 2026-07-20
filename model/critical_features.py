from demo.train_demo_model import train_model
from data.load_data import load_data
import pandas as pd


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    mod1 = train_model()

    importances = mod1.model.feature_importances_  # accessing the raw sklearn model inside your wrapper
    feature_columns = x_train.columns

    for feature, importance in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        print(f"{feature}: {importance:.4f}")
        
    df = pd.read_csv("data/loan_approval_data.csv")
    print(df["Credit_Score"].describe())