# --- Imports ---
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preprocessing
from src.preprocessing.preprocessing import (
    load_or_fetch_data,
    prepare_dataframes,
    feature_engineering,
    final_cleaning,
    save_clean_data
)

# Encoding
from src.modeling.encoding import (
    label_encode_columns,
    one_hot_encode_columns
)

# Modeling
from src.modeling.model_training import (
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_decision_tree,
    apply_smote
)
from src.modeling.hyperparameter_search import tune_random_forest
from src.modeling.stacking import train_stacking, evaluate_stacking

# Evaluation
from src.modeling.evaluation import (
    plot_roc_curve,
    plot_feature_importances,
    shap_summary_plot
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

    # Data preparation steps
    animal_df = prepare_dataframes(df_intake, df_outcome)
    animal_df = feature_engineering(animal_df)
    df = final_cleaning(animal_df)
    save_clean_data(df, filename=os.path.join(DATA_FOLDER, "animal_df.csv"))

    print(f"✅ Preprocessing complete. Dataset shape: {df.shape}")

    # --- Step 2: Causal Inference ---
    print("🧠 Running causal inference analysis...")

    df1 = pd.read_csv("data/animal_df.csv")

    treatment = "age_days_outcome"
    outcome = "los_at_shelter"
    common_causes = ["animal_type", "breed_type", "intake_condition_group"]

    causal_estimate, refutation_placebo, refutation_random, refutation_subset = causal_inference_pipeline(
        df1, treatment, outcome, common_causes
    )

    print(f"✅ Final Causal Effect Estimate: {causal_estimate.value}")

    # --- Step 3: Encoding ---
    print("🔤 Applying encoding...")
    
    label_encode_cols = [
        'age_group_intake', 'breed_type', 'color_group', 
        'intake_condition_group', 'animal_type', 'month_of_outcome'
    ]
    df_encoded, label_encoders = label_encode_columns(df, label_encode_cols)

    # --- Step 4: Prepare for modeling ---
    print("🛠 Preparing train/test data...")

    X = df_encoded.drop(columns=['outcome_group','name_intake', 'name_outcome'])
    y = df_encoded['outcome_group']

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print('X_train_full shape = ', X_train_full.shape)
    
    # Apply SMOTE-Tomek
    print("🔧 Applying SMOTE-Tomek...")
    X_train_bal, y_train_bal = apply_smote(X_train_full, y_train_full)
    print('X_train_bal shape = ', X_train_bal.shape)

    # --- Step 5: Modeling ---
    print("🌳 Training Random Forest...")

    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    rf_model = train_random_forest(X_train_bal, y_train_bal, best_params)

    # --- Step 6: Stacking ---
    print("⚡ Training stacking model...")

    base_models = [("rf", rf_model)]
    meta_model = LogisticRegression()
    stacking_model = train_stacking(meta_model, base_models, X_train_bal, y_train_bal)

    print("✅ Evaluating stacking model...")
    stacking_accuracy = evaluate_stacking(stacking_model, X_test, y_test)
    print(f"🏆 Stacking Model Accuracy: {stacking_accuracy:.4f}")

    # --- Step 7: Clustering ---
    print("🔍 Running K-Prototypes clustering...")

    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "color_group", "breed_type"]

    clusters, kproto_model = run_kprototypes(X_train_full, numerical_features, categorical_features)

    print("\n🎯 FULL PIPELINE SUCCESSFULLY COMPLETED. 🚀")

# --- Entry Point ---
if __name__ == "__main__":
    main()
