# src/main.py

# --- Imports ---

# Preprocessing
from preprocessing.preprocessing import (
    fetch_data_from_web,
    prepare_dataframes,
    feature_engineering,
    final_cleaning,
    save_clean_data
)

# Encoding
from modeling.encoding import (
    label_encode_columns,
    one_hot_encode_columns
)

# Causal Inference
from causal_inference.causal_analysis import causal_inference_pipeline

# Modeling
from modeling.model_training import (
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_decision_tree,
    apply_smote
)
from modeling.hyperparameter_search import tune_random_forest
from modeling.stacking import train_stacking, evaluate_stacking

# Clustering
from clustering.kprototypes_clustering import run_kprototypes

# Evaluation
from modeling.evaluation import plot_roc_curve, plot_feature_importances, shap_summary_plot

# Other
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os

def main():
    # 1. Download and preprocess data
    print("Downloading and preprocessing data...")

    # Base URLs
    base_url_outcome = "https://data.austintexas.gov/resource/9t4d-g238.csv"
    base_url_intake = "https://data.austintexas.gov/resource/wter-evkm.csv"

    data_dir = "Group_Project_IS2/data"
    intake_path = os.path.join(data_dir, "intake_raw.csv")
    outcome_path = os.path.join(data_dir, "outcome_raw.csv")

    # Check if data already exists
    if os.path.exists(intake_path) and os.path.exists(outcome_path):
        print("Found existing intake and outcome CSVs, loading...")
        df_intake = pd.read_csv(intake_path)
        df_outcome = pd.read_csv(outcome_path)
    else:
        print("No existing CSVs found, fetching from API...")
        df_intake = fetch_data_from_web(base_url_intake)
        df_outcome = fetch_data_from_web(base_url_outcome)

        os.makedirs(data_dir, exist_ok=True)
        df_intake.to_csv(intake_path, index=False)
        df_outcome.to_csv(outcome_path, index=False)
        print("Data fetched and saved.")

    # Merge and preprocess
    animal_df = prepare_dataframes(df_intake, df_outcome)
    animal_df = feature_engineering(animal_df)
    df = final_cleaning(animal_df)

    # Save cleaned dataset
    save_clean_data(df, filename=os.path.join(data_dir, "animal_df.csv"))

    print(f"Preprocessing complete. Dataset shape: {df.shape}")

    # 2. Encoding
    print("Applying encoding...")

    label_encode_cols = ['age_group_intake', 'breed_type', 'color_group', 'intake_condition_group', 'animal_type', 'month_of_outcome']
    
    df_encoded, label_encoders = label_encode_columns(df, label_encode_cols)
    # Optional: One-Hot Encoding if needed
    # one_hot_cols = ['animal_type', 'color_group']
    # df_encoded, onehot_encoder = one_hot_encode_columns(df_encoded, one_hot_cols)

    # 3. Causal Inference
    print("Running causal inference analysis...")

    treatment = "age_days_outcome"
    outcome = "los_at_shelter"
    common_causes = [ "animal_type", "breed_type", "intake_condition_group"]

    # Prepare numeric-only data
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if outcome not in numeric_cols:
        numeric_cols.append(outcome)

    df_causal = df_encoded[numeric_cols]

    causal_estimate = causal_inference_pipeline(df_causal, treatment, outcome, common_causes)

    # 4. Define Features and Target
    print("Preparing train/test data...")

    X = df_encoded.drop(columns=[outcome])
    y = df_encoded[outcome]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Apply SMOTE-Tomek
    print("Applying SMOTE-Tomek on training data for modeling...")
    X_train_bal, y_train_bal = apply_smote(X_train_full, y_train_full)

    # 6. Train Random Forest
    print("Training Random Forest model...")
    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    rf_model = train_random_forest(X_train_bal, y_train_bal, best_params)

    # 7. Stacking
    print("Training stacking model...")
    base_models = [("rf", rf_model)]
    meta_model = LogisticRegression()
    stacking_model = train_stacking(meta_model, base_models, X_train_bal, y_train_bal)

    # 8. Evaluate Stacking
    print("Evaluating stacking model...")
    stacking_accuracy = evaluate_stacking(stacking_model, X_test, y_test)
    print(f"Stacking Model Accuracy: {stacking_accuracy:.4f}")

    # 9. Clustering
    print("Running K-Prototypes clustering...")
    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "color_group", "breed_type"]
    clusters, kproto_model = run_kprototypes(X_train_full, numerical_features, categorical_features)

    # 10. Optional Visualizations
    print("Plotting feature importances and SHAP...")
    # plot_feature_importances(rf_model, X_train_bal)
    # shap_summary_plot(rf_model, X_train_bal)

    print("\nFULL PIPELINE SUCCESSFULLY COMPLETED.")

if __name__ == "__main__":
    main()
