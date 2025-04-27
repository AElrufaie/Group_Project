import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek

# Apply SMOTE-Tomek for Balancing
def apply_smote(X, y):
    """
    Apply SMOTE-Tomek resampling to balance classes.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        X_res (pd.DataFrame): Resampled features.
        y_res (pd.Series): Resampled target.
    """
    smote_tomek = SMOTETomek(random_state=42)
    X_res, y_res = smote_tomek.fit_resample(X, y)
    return X_res, y_res

# Train Random Forest
def train_random_forest(X_train, y_train, best_params):
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train XGBoost
def train_xgboost(X_train, y_train, best_params):
    model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Train LightGBM
def train_lightgbm(X_train, y_train, best_params):
    model = LGBMClassifier(**best_params, random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Train CatBoost
def train_catboost(X_train, y_train, best_params):
    model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    return model

# Train Decision Tree
def train_decision_tree(X_train, y_train, best_params):
    model = DecisionTreeClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model
