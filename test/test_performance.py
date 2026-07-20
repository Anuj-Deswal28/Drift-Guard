from performance.metrics import compute_accuracy, compute_auc

def test_accuracy_known_case():
    predictions = [1, 0, 1, 0]
    true_labels = [1, 0, 0, 0]
    
    accuracy = compute_accuracy(true_labels, predictions)
    
    assert accuracy == 0.75

def test_accuracy_perfect_case():
    predictions = [1, 1, 0, 0]
    true_labels = [1, 1, 0, 0]
    
    accuracy = compute_accuracy(true_labels, predictions)
    
    assert accuracy == 1  
    
def test_auc_perfect_case():
    probabilities = [0.9, 0.8, 0.2, 0.1] 
    true_labels =    [1,   1,   0,   0]   
    
    auc = compute_auc(true_labels, probabilities)
    
    assert auc == 1.0
    
