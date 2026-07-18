import numpy as np

def psi(base_feature_data, comparison_feature_data, num_bins=10):

    percentiles = np.linspace(0, 100, num=11) 
    bin_edges = np.percentile(base_feature_data, percentiles)    

    # Step 2 (coming next): count how many baseline vs comparison values fall in each bin
    base_counts, _ = np.histogram(base_feature_data, bins=bin_edges)
    comparison_counts, _ = np.histogram(comparison_feature_data, bins=bin_edges)

    # Step 3 (coming next): convert counts to percentages, apply the PSI formula
    epsilon = 0.0001
    base_percentages = base_counts / base_counts.sum() + epsilon
    comparison_percentages = comparison_counts / comparison_counts.sum() + epsilon
    
    psi_score = np.sum((comparison_percentages - base_percentages) * np.log(comparison_percentages / base_percentages))

    return psi_score


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    baseline = pd.Series(np.random.normal(loc=50, scale=10, size=500))   # centered around 50
    no_drift = pd.Series(np.random.normal(loc=50, scale=10, size=200))   # same distribution
    drifted  = pd.Series(np.random.normal(loc=80, scale=10, size=200))   # shifted way up

    print("PSI (no real drift):", psi(baseline, no_drift))
    print("PSI (real drift):", psi(baseline, drifted))