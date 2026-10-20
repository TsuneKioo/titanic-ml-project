
from sklearn.dummy import DummyClassifier

def get_baseline_model(seed = 42):
    params = {
        "strategy": "most_frequent",
        "random_state": seed
    }
    model = DummyClassifier(**params)
    return model, params