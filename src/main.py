# --- Imports ---
import mlflow
import pandas as pd
import numpy as np
import optuna
import os
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


# Try checking for GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU is NOT available. Running on CPU.")
except ImportError:
    print("⚠️ Torch not installed. Cannot auto-check GPU availability.")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preprocessing
from src.preprocessing.preprocessing import (
    load_or_fetch_data, prepare_dataframes, feature_engineering, final_cleaning, save_clean_data
)

# Encoding
from src.modeling.encoding import label_encode_columns

# Modeling
from src.modeling.model_training import (
    train_random_forest, train_xgboost, train_catboost, apply_smote
)
from src.modeling.hyperparameter_search import (
    tune_random_forest, tune_catboost, tune_xgboost
)
from src.modeling.stacking import (
        get_meta_models,
        train_and_evaluate_stacking_models,
        get_base_model_combinations
    )

# Clustering
from src.clustering.kprototypes_clustering import run_kprototypes

# Causal Inference
from src.causal_inference.causal_analysis import causal_inference_pipeline



# --- Global Variables ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

# --- Main Function ---
def main():
    print("🚀 Downloading and preprocessing data...")

    # Base URLs
    base_url_1 = "https://data.austintexas.gov/resource/9t4d-g238.csv"
    base_url_2 = "https://data.austintexas.gov/resource/wter-evkm.csv"

    # Load or fetch data
    df_intake, df_outcome = load_or_fetch_data(DATA_FOLDER, base_url_1, base_url_2)
    animal_df = prepare_dataframes(df_intake, df_outcome)
    animal_df = feature_engineering(animal_df)
    df = final_cleaning(animal_df)
    save_clean_data(df, filename=os.path.join(DATA_FOLDER, "animal_df.csv"))

    print(f"✅ Preprocessing complete. Dataset shape: {df.shape}")

    # --- Step 2: Causal Inference ---
    print("🧠 Running causal inference analysis...")

    df1 = pd.read_csv(os.path.join(DATA_FOLDER, "animal_df.csv"))

    treatment = "age_days_outcome"
    outcome = "los_at_shelter"
    common_causes = ["animal_type", "breed_type", "intake_condition_group"]

    # Uncomment if you want causal results
    causal_estimate, refutation_placebo, refutation_random, refutation_subset = causal_inference_pipeline(df1, treatment, outcome, common_causes)
    print(f"✅ Final Causal Effect Estimate: {causal_estimate.value}")

    # --- Step 3: Encoding ---
    print("🔤 Applying encoding...")
    label_encode_cols = ['age_group_intake', 'breed_type', 'color_group', 'intake_condition_group', 'animal_type', 'month_of_outcome']
    df_encoded, label_encoders = label_encode_columns(df, label_encode_cols)

    # --- Step 4: Prepare for modeling ---
    print("🛠 Preparing train/test data...")
    X = df_encoded.drop(columns=['outcome_group', 'name_intake', 'name_outcome'])
    y = df_encoded['outcome_group']

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('X_train_full shape = ', X_train_full.shape)

    print("🔧 Applying SMOTE-Tomek...")
    X_train_bal, y_train_bal = apply_smote(X_train_full, y_train_full)
    print('X_train_bal shape = ', X_train_bal.shape)

    # --- Step 5: Prepare Labels ---
    print("🎯 Encoding labels for Boosting Models...")
    label_encoder = LabelEncoder()
    y_train_bal_encoded = label_encoder.fit_transform(y_train_bal)
    y_test_encoded = label_encoder.transform(y_test)

   # --- Step 6: Hyperparameter Tuning ---
    print("🔍 Starting Hyperparameter Tuning...")

    best_params = {}

    # Tune XGBoost
    print("🎯 Tuning XGBoost...")
    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(lambda trial: tune_xgboost(trial, X_train_bal, X_test, y_train_bal_encoded, y_test_encoded), n_trials=10)
    best_params['xgb'] = xgb_study.best_params
    print(f"✅ Best XGB params: {best_params['xgb']}")

    # Tune CatBoost
    print("🎯 Tuning CatBoost...")
    cat_study = optuna.create_study(direction="maximize")
    cat_study.optimize(lambda trial: tune_catboost(trial, X_train_bal, X_test, y_train_bal_encoded, y_test_encoded), n_trials=10)
    best_params['cat'] = cat_study.best_params
    print(f"✅ Best CatBoost params: {best_params['cat']}")

    # Tune Random Forest
    print("🎯 Tuning Random Forest...")
    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(lambda trial: tune_random_forest(trial, X_train_bal, X_test, y_train_bal_encoded, y_test_encoded), n_trials=10)
    best_params['rf'] = rf_study.best_params
    print(f"✅ Best RF params: {best_params['rf']}")

    # --- Step 7: Stacking ---
    print("⚡ Training stacking model with XGB + CAT + RF...")

    # Get meta models
    meta_models = get_meta_models()

    # Get base model combinations (only XGB, CAT, RF)
    base_model_combinations = get_base_model_combinations(best_params)

    # Train and evaluate stacking models
    train_and_evaluate_stacking_models(
        meta_models,
        base_model_combinations,
        X_train_bal,
        y_train_bal_encoded,
        X_test,
        y_test_encoded
    )



  # --- Step 8: Clustering ---
    print("🔍 Running K-Prototypes clustering on full clean dataset...")

    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "has_name", "age_group_intake", 
                        "month_of_outcome", "is_fixed", "breed_type", 
                        "color_group", "intake_condition_group"]

    clusters, kproto_model = run_kprototypes(df, numerical_features, categorical_features)

    # Save the clustered dataset
    df['kprototypes_cluster'] = clusters
    df.to_csv(os.path.join(DATA_FOLDER, "animal_df_with_clusters.csv"), index=False)
    print("✅ Clustered data saved as 'animal_df_with_clusters.csv'")

    # ✅ Compute silhouette score manually here
    from sklearn.metrics import silhouette_score

    silhouette = silhouette_score(
        df[numerical_features],  # only numerical features
        clusters,
        metric="euclidean"
    )

    print(f"✅ Silhouette Score for Clustering: {silhouette:.4f}")

    print("\n🎯 FULL PIPELINE SUCCESSFULLY COMPLETED. 🚀")

# --- Entry Point ---
if __name__ == "__main__":
    main()
