# src/modeling/model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek

# Import your MLflow manager
from src.mlflow_management import mlflow_manager

# Apply SMOTE-Tomek for Balancing
def apply_smote(X, y):
    """
    Apply SMOTE-Tomek resampling to balance classes.
    """
    smote_tomek = SMOTETomek(random_state=42)
    X_res, y_res = smote_tomek.fit_resample(X, y)
    return X_res, y_res

# Train Random Forest
def train_random_forest(X_train, y_train, best_params):
    mlflow_manager.start_run(run_name="RandomForest_Training")
    
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    mlflow_manager.log_params(best_params)
    mlflow_manager.log_model(model, model_name="random_forest_model", X_train=X_train, y_train=y_train)
    
    mlflow_manager.end_run()
    return model

# Train XGBoost
def train_xgboost(X_train, y_train, best_params):
    mlflow_manager.start_run(run_name="XGBoost_Training")
    
    model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    mlflow_manager.log_params(best_params)
    mlflow_manager.log_model(model, model_name="xgboost_model", X_train=X_train, y_train=y_train)
    
    mlflow_manager.end_run()
    return model

# Train LightGBM
def train_lightgbm(X_train, y_train, best_params):
    mlflow_manager.start_run(run_name="LightGBM_Training")
    
    model = LGBMClassifier(**best_params, random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    mlflow_manager.log_params(best_params)
    mlflow_manager.log_model(model, model_name="lightgbm_model", X_train=X_train, y_train=y_train)
    
    mlflow_manager.end_run()
    return model

# Train CatBoost
def train_catboost(X_train, y_train, best_params):
    mlflow_manager.start_run(run_name="CatBoost_Training")
    
    model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
    model.fit(X_train, y_train)

    mlflow_manager.log_params(best_params)
    mlflow_manager.log_model(model, model_name="catboost_model", X_train=X_train, y_train=y_train)
    
    mlflow_manager.end_run()
    return model

# Train Decision Tree
def train_decision_tree(X_train, y_train, best_params):
    mlflow_manager.start_run(run_name="DecisionTree_Training")
    
    model = DecisionTreeClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    mlflow_manager.log_params(best_params)
    mlflow_manager.log_model(model, model_name="decision_tree_model", X_train=X_train, y_train=y_train)
    
    mlflow_manager.end_run()
    return model

# For Testing
def train_model(X_train, y_train, X_val, y_val, params):
    """Train a model and return it."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
