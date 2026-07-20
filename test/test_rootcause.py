from root_cause.correlation import compute_root_cause_correlation

def test_perfect_negative_correlation():
    drift_series = {
        "Feature_A": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    performance_series = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    result = compute_root_cause_correlation(drift_series, performance_series)
    
    correlation = result["Feature_A"]["correlation"]
    
    assert correlation < -0.99  