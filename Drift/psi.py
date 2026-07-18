import numpy as np
import pandas as pd
from Ingestion.query import get_feature_data
from scipy.stats import ks_2samp

def psi(base_feature_data, comparison_feature_data, num_bins=10):

    percentiles = np.linspace(0, 100, num=11) 
    bin_edges = np.percentile(base_feature_data, percentiles)    

    
    base_counts, _ = np.histogram(base_feature_data, bins=bin_edges)
    comparison_counts, _ = np.histogram(comparison_feature_data, bins=bin_edges)

    
    epsilon = 0.0001
    base_percentages = base_counts / base_counts.sum() + epsilon
    comparison_percentages = comparison_counts / comparison_counts.sum() + epsilon
    
    psi_score = np.sum((comparison_percentages - base_percentages) * np.log(comparison_percentages / base_percentages))

    return psi_score

def compute_all_feature_psi(baseline_start, baseline_end, comparison_start, comparison_end):
    feature_columns = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    psi_scores = {}
    
    for feature in feature_columns:
        
        base_data = get_feature_data(feature, baseline_start, baseline_end)
        comparison_data = get_feature_data(feature, comparison_start, comparison_end)
        
        psi_score = psi(base_data, comparison_data) 
        psi_scores[feature] = psi_score  
    
    return psi_scores



def ks_test(base_feature_data, comparison_feature_data):
    ks_statistic, p_value = ks_2samp(base_feature_data, comparison_feature_data)  
    return ks_statistic, p_value  

def compute_all_feature_ks(baseline_start, baseline_end, comparison_start, comparison_end):
    feature_columns = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    ks_scores = {}
    
    for feature in feature_columns:
        
        base_data = get_feature_data(feature, baseline_start, baseline_end)
        comparison_data = get_feature_data(feature, comparison_start, comparison_end)
        
        statistic , p_value = ks_test(base_data, comparison_data) 
        ks_scores[feature] = {"statistic": statistic, "p_value": p_value}
    
    return ks_scores
    



if __name__ == "__main__":
    ks_scores = compute_all_feature_ks("2026-01-01", "2026-01-04", "2026-01-06", "2026-01-07")
    print(ks_scores)