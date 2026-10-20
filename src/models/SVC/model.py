from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_svc_pipeline(params = None, seed = 42):
    if params is None:
        params = {}

    pipeline = make_pipeline(
        StandardScaler(), 
        SVC(**params, probability=True, random_state=seed)
        )
    
    return pipeline