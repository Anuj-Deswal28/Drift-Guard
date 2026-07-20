import numpy as np
import pandas as pd
from Drift.psi import ks_test  

def test_ks_no_drift():
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    no_drift = pd.Series(np.random.normal(loc=50, scale=10, size=200))
    
    statistic, p_value = ks_test(baseline, no_drift)
    
    assert p_value > 0.05  

def test_ks_with_drift():
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    drifted = pd.Series(np.random.normal(loc=80, scale=10, size=200))
    
    statistic, p_value = ks_test(baseline, drifted)
    
    assert p_value < 0.05  