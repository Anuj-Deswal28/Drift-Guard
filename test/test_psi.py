import numpy as np
import pandas as pd
from Drift.psi import psi

def test_psi_no_drift():
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    no_drift = pd.Series(np.random.normal(loc=50, scale=10, size=200))
    
    score = psi(baseline, no_drift)
    
    assert score < 0.1  

def test_psi_with_drift():
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(loc=50, scale=10, size=500))
    drifted = pd.Series(np.random.normal(loc=80, scale=10, size=200))
    
    score = psi(baseline, drifted)
    
    assert  score >= 0.25  