from scipy.stats import pearsonr
from Drift.psi import compute_all_feature_psi
from datetime import datetime, timedelta
from performance.metrics import compute_accuracy, compute_auc
from Ingestion.query import get_feature_data


def generate_root_cause_report(baseline_start, baseline_end):
    day_starts, day_ends = get_time_series()
    drift_series = get_drift_timeseries(baseline_start, baseline_end, day_starts, day_ends)
    performance_series = get_performance_timeseries(day_starts, day_ends)
    root_cause = compute_root_cause_correlation(drift_series, performance_series)
    
    sorted_features = sorted(root_cause.items(), key=lambda x: x[1]["correlation"])
    return sorted_features

def get_drift_timeseries(baseline_start, baseline_end, day_starts, day_ends):
    feature_columns = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    drift_series = {feature: [] for feature in feature_columns}  
    
    for start, end in zip(day_starts, day_ends):
        daily_psi = compute_all_feature_psi(baseline_start, baseline_end, start, end)
        for feature in feature_columns:
            drift_series[feature].append(daily_psi[feature])
            
    return drift_series

def get_time_series(start_day=3, end_day=9):
    start_anchor = datetime(2026, 1, 1)

    day_starts = [(start_anchor + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(start_day, end_day + 1)]
    day_ends = [(start_anchor + timedelta(days=d+1)).strftime('%Y-%m-%d') for d in range(start_day, end_day + 1)]

    return day_starts, day_ends



def get_performance_timeseries(day_starts, day_ends):
    accuracy_series = []
    
    for start, end in zip(day_starts, day_ends):
        data = get_feature_data(["Prediction", "true_lable"], start, end)
        acc = compute_accuracy(data["true_lable"], data["Prediction"]) 
        accuracy_series.append(acc)
    
    return accuracy_series

def compute_root_cause_correlation(drift_series, performance_series):
    correlations = {}
    
    for feature, drift_values in drift_series.items():
        correlation, p_value = pearsonr(drift_values,performance_series) 
        correlations[feature] = {"correlation": correlation, "p_value": p_value}
    
    return correlations


if __name__ == "__main__":
    report = generate_root_cause_report("2026-01-01", "2026-01-04")
    for feature, stats in report:
        print(feature, stats)