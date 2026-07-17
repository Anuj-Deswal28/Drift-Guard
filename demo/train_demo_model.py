from data.load_data import load_data
from model.wrapper import Drift_guard
from sklearn.ensemble import RandomForestClassifier

def train_model():
    model = RandomForestClassifier(random_state = 32, max_depth = 10, min_samples_leaf=3)

    x_train, x_test, y_train, y_test = load_data()
    model.fit(x_train,y_train)

    mod1 = Drift_guard(model)
    
    return mod1

if __name__ == "__main__" :
    x_train, x_test, y_train, y_test = load_data()
    mod1 = train_model()
    print(mod1.pred(x_test)[:5])
    print(mod1.pred_prob(x_test)[:5])