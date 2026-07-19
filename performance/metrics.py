from sklearn.metrics import accuracy_score, roc_auc_score
from Ingestion.query import get_feature_data

def compute_accuracy(true_labels, predictions):
    accuracy = accuracy_score(true_labels,predictions)
    return accuracy

def compute_auc(true_labels, probabilities):
    auc = roc_auc_score(true_labels,probabilities)
    return auc

def compute_performance_report(baseline_start, baseline_end, comparison_start, comparison_end):
    baseline_data = get_feature_data(["Prediction", "Probablity", "true_lable"], baseline_start, baseline_end)
    comparison_data = get_feature_data(["Prediction", "Probablity", "true_lable"], comparison_start, comparison_end)
    
    result = {}
    
    result["base_line_acc"] = compute_accuracy(baseline_data["true_lable"],baseline_data["Prediction"])
    result["base_line_auc"] = compute_auc(baseline_data["true_lable"],baseline_data["Probablity"])
    
    result["comparision_acc"] = compute_accuracy(comparison_data["true_lable"], comparison_data["Prediction"])
    result["comparision_auc"] = compute_auc(comparison_data["true_lable"], comparison_data["Probablity"])
    
    return result 

if __name__ == "__main__":
    report = compute_performance_report("2026-01-01", "2026-01-04", "2026-01-06", "2026-01-07")
    print(report)