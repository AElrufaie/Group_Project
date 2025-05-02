import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# ---------- Model-based Data Drift Utilities ----------

def split_reference_current(df: pd.DataFrame, ref_frac: float = 0.5, random_state: int = 42):
    """
    Split DataFrame into reference and current datasets.
    Returns ref_df, curr_df, and combined DataFrame with labels.
    """
    ref = df.sample(frac=ref_frac, random_state=random_state).reset_index(drop=True)
    curr = df.drop(ref.index).sample(frac=ref_frac, random_state=random_state).reset_index(drop=True)
    ref['__drift_label__'] = 0
    curr['__drift_label__'] = 1
    combined = pd.concat([ref, curr], ignore_index=True)
    return ref, curr, combined


def detect_model_drift(
    combined: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    threshold: float = 0.55,
    test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    """
    Train a classifier to distinguish reference (0) vs current (1).
    Use AUC or accuracy to decide drift: if metric > threshold, drift detected.
    Returns {'metric':value, 'drift':bool, 'threshold':threshold}.
    """
    # Prepare features
    X_num = combined[numerical_features].fillna(0)
    X_cat = combined[categorical_features].fillna('NA')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_enc = encoder.fit_transform(X_cat)
    X = np.hstack([X_num.values, X_cat_enc])
    y = combined['__drift_label__'].values

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predict probabilities
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
        metric_val = roc_auc_score(y_test, y_prob)
        metric_name = 'roc_auc'
    else:
        y_pred = clf.predict(X_test)
        metric_val = accuracy_score(y_test, y_pred)
        metric_name = 'accuracy'

    drift_flag = metric_val > threshold
    return {'metric_name': metric_name, 'metric_value': metric_val, 'threshold': threshold, 'drift': drift_flag}

# ---------- Data Drift Workflow Function (Model-based) ----------
inject_params = {
    'numeric': {
        'age_days_outcome': (5, 2),
        'los_at_shelter':  (0, 1)
    },
    'categorical': {

        'breed_type': ('NEW', 0.3)
    }
}

def run_model_based_drift_workflow(
    df: pd.DataFrame,
    numerical_features: list,
    categorical_features: list,
    threshold: float = 0.55,
    inject: bool = True,
    inject_params = inject_params
) -> dict:
    """
    1. Split reference/current, combine with labels
    2. Optional: inject drift for demo
    3. Detect drift via model-based method
    4. Print results
    Returns drift_info dict.
    """
    # 1. Split and combine
    ref, curr, combined = split_reference_current(df)

    # 2. Optional injection (reuse previous utilities)
    if inject and inject_params is not None:
        from scipy.stats import norm
        for feat in inject_params.get('numeric', []):
            loc, scale = inject_params['numeric'][feat]
            noise = np.random.normal(loc=loc, scale=scale, size=curr.shape[0])
            curr[feat] += noise
        for feat in inject_params.get('categorical', {}):
            new_cat, prob = inject_params['categorical'][feat]
            mask = np.random.rand(curr.shape[0]) < prob
            curr.loc[mask, feat] = new_cat
        combined = pd.concat([ref.assign(__drift_label__=0), curr.assign(__drift_label__=1)], ignore_index=True)

    # 3. Detect model-based drift
    drift_info = detect_model_drift(
        combined, numerical_features, categorical_features, threshold=threshold
    )

    # 4. Print summary
    print("=== Model-based Drift Detection ===")
    print(f"Metric: {drift_info['metric_name']} = {drift_info['metric_value']:.4f}")
    print(f"Threshold: {drift_info['threshold']}")
    print("Drift detected!" if drift_info['drift'] else "No drift detected.")

    return drift_info

# ---------- Main Workflow ----------

def main():
    df = pd.read_csv(r"C:\Users\Kevin\Desktop\MLE Project - Fatih\Group_Project_IS2\data\animal_df_clean.csv")
    numerical = ["age_days_outcome", "los_at_shelter"]
    categorical = ["age_group_intake", "is_fixed", "breed_type", "color_group", "intake_condition_group", "month_of_outcome"]

    # Example: model-based drift without injection
    run_model_based_drift_workflow(
        df,
        numerical_features=numerical,
        categorical_features=categorical,
        threshold=0.55,
        inject=True
    )

if __name__ == "__main__":
    main()
