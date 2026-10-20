from xgboost import XGBClassifier

def get_xgb_model(params = None, seed = 42):
    if params is None:
        params = {}

    model = XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric="auc",
                tree_method="hist",
                device="cuda",
                random_state = seed
            )
    return model
