import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.models.Baseline.model import get_baseline_model
from src.models.SVC.model import get_svc_pipeline
from src.models.XGBoost.model import get_xgb_model

import yaml
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
config_path = root_dir / "configs" / "config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    configs = yaml.safe_load(f)

SEED = configs["data"]["seed"]

PROJECT_ROOT = Path(__file__).resolve().parent
OPTUNA_DIR = PROJECT_ROOT / "artifacts" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = f"sqlite:///{OPTUNA_DIR / 'studies.db'}"

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('titanic')

X_train = pd.read_parquet("data/processed/train_data.parquet")
X_test = pd.read_parquet("data/processed/test_data.parquet")

y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()

with mlflow.start_run(run_name='baseline_most_frequent'):

    dummy, params = get_baseline_model(seed = SEED)
    dummy.fit(X_train, y_train)

    preds = dummy.predict(X_test)
    probs = dummy.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_roc_auc', roc_auc)

    mlflow.sklearn.log_model(dummy, name='baseline')


optuna.logging.set_verbosity(optuna.logging.WARNING)

with mlflow.start_run(run_name='svc_optuna'):

    def objective(trial):
        params = {
            'C' : trial.suggest_float("C", 0.1, 100, log = True),
            'gamma' : trial.suggest_float("gamma", 0.001, 0.1, log = True),
            'kernel' : 'rbf'
        }

        clf = get_svc_pipeline(params = params, seed = SEED)

        cv = StratifiedKFold(n_splits=5, shuffle= True, random_state=SEED)
        
        roc_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc').mean()

        return roc_auc

    study_svc = optuna.create_study(study_name='svc', direction="maximize", sampler = optuna.samplers.TPESampler(seed=SEED),
                                storage=STORAGE, load_if_exists=True)

    study_svc.optimize(objective, n_trials=50)

    best_params = study_svc.best_params
    best_score = study_svc.best_value

    print("Best hyperparameters:", study_svc.best_params)
    print("Best roc_auc:", study_svc.best_value)

    best_svc_model = get_svc_pipeline(params=study_svc.best_params, seed=SEED)
    best_svc_model.fit(X_train, y_train)
    preds = best_svc_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_roc_auc', best_score)

    mlflow.sklearn.log_model(best_svc_model, name='svc')


with mlflow.start_run(run_name='xgboost_optuna'):

    def objective(trial):

        params = {
        'learning_rate' : trial.suggest_float("learning_rate", 0.01, 0.1),
        'max_depth' : trial.suggest_int("max_depth", 2, 4),
        'n_estimators' : trial.suggest_int("n_estimators", 50, 100),
        'subsample' : trial.suggest_float("subsample", 0.6, 0.8, step = 0.1),
        'colsample_bytree' : trial.suggest_float("colsample_bytree", 0.4, 0.7, step=0.1),
        'min_child_weight' : trial.suggest_int('min_child_weight', 4, 10),
        'gamma' : trial.suggest_float("gamma", 0, 5),
        'reg_alpha' : trial.suggest_float("reg_alpha", 0.0, 1.0),
        'reg_lambda' : trial.suggest_float( "reg_lambda", 1.0, 10.0),
        }

        clf = get_xgb_model(params = params, seed = SEED)

        roc_auc = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()

        return roc_auc

    xgb_study = optuna.create_study(study_name='xgb', direction="maximize", sampler = optuna.samplers.TPESampler(seed=SEED),
                                storage=STORAGE, load_if_exists=True)

    xgb_study.optimize(objective, n_trials=50)

    best_params = xgb_study.best_params
    best_score = xgb_study.best_value

    print("Best hyperparameters:", xgb_study.best_params)
    print("Best roc_auc:", xgb_study.best_value)

    best_xgb_model = get_xgb_model(params = xgb_study.best_params, seed = SEED)
    best_xgb_model.fit(X_train, y_train)
    preds = best_xgb_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_roc_auc', best_score)

    trusted_types = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

    mlflow.sklearn.log_model(best_xgb_model, name='xgboost', skops_trusted_types=trusted_types)