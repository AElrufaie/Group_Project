# src/modeling/hyperparameter_search.py

import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Import your MLflow manager
from src.mlflow_management import mlflow_manager

# --- Random Forest Tuning (CPU) ---
def tune_random_forest(trial, X_train, X_test, y_train, y_test):
    mlflow_manager.start_run(run_name="RandomForest_Tuning")

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
    }

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow_manager.log_params(params)
    mlflow_manager.log_metrics({"accuracy": accuracy})
    mlflow_manager.end_run()

    return accuracy

# --- XGBoost Tuning (GPU) ---
def tune_xgboost(trial, X_train, X_test, y_train, y_test):
    mlflow_manager.start_run(run_name="XGBoost_Tuning")

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'tree_method': 'hist',           # Use histogram-based algorithm on CPU
        'predictor': 'cpu_predictor',    # Force CPU prediction
        'device': 'cpu'                  # Make it very explicit
                     # Use GPU 0
    }

    model = XGBClassifier(**params, use_label_encoder=False, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow_manager.log_params(params)
    mlflow_manager.log_metrics({"accuracy": accuracy})
    mlflow_manager.end_run()

    return accuracy

# --- LightGBM Tuning (GPU) ---
def tune_lightgbm(trial, X_train, X_test, y_train, y_test):
    mlflow_manager.start_run(run_name="LightGBM_Tuning")

    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(set(y_train)),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'device': 'cpu'      # GPU acceleration
    }

    model = LGBMClassifier(**params, random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow_manager.log_params(params)
    mlflow_manager.log_metrics({"accuracy": accuracy})
    mlflow_manager.end_run()

    return accuracy

# --- Decision Tree Tuning (CPU) ---
def tune_decision_tree(trial, X_train, X_test, y_train, y_test):
    mlflow_manager.start_run(run_name="DecisionTree_Tuning")

    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
    }

    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow_manager.log_params(params)
    mlflow_manager.log_metrics({"accuracy": accuracy})
    mlflow_manager.end_run()

    return accuracy

# --- CatBoost Tuning (GPU) ---
def tune_catboost(trial, X_train, X_test, y_train, y_test):
    mlflow_manager.start_run(run_name="CatBoost_Tuning")

    params = {
        'loss_function': 'MultiClass',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'task_type': 'CPU'   # GPU acceleration
    }

    model = CatBoostClassifier(**params, random_seed=42, verbose=0, eval_metric='Accuracy')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow_manager.log_params(params)
    mlflow_manager.log_metrics({"accuracy": accuracy})
    mlflow_manager.end_run()

    return accuracy
