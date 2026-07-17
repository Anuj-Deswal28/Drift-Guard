class Drift_guard():
    def __init__(self,mod):
        self.model = mod
        
    def pred(self,x):
        results = self.model.predict(x)
        return results
        
    def pred_prob(self,x):
        if hasattr(self.model, "predict_proba"):
            result = self.model.predict_proba(x)
            return result
        return self.pred(x)