# src/main.py

# --- Imports ---

# Preprocessing
from preprocessing.preprocessing import (
    download_data,
    merge_latest_intake_outcome,
    drop_unnecessary_columns,
    create_features,
    simplify_features,
    final_cleanup
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

def main():
    # 1. Download and preprocess data step-by-step
    print("Downloading and preprocessing data...")

    base_url_1 = "https://data.austintexas.gov/resource/9t4d-g238.csv"
    base_url_2 = "https://data.austintexas.gov/resource/wter-evkm.csv"

    df_intake, df_outcome = download_data(base_url_1, base_url_2)
    animal_df = merge_latest_intake_outcome(df_intake, df_outcome)
    animal_df = drop_unnecessary_columns(animal_df)
    animal_df = create_features(animal_df)
    animal_df = simplify_features(animal_df)
    df = final_cleanup(animal_df)

    print(f"Preprocessing complete. Dataset shape: {df.shape}")

    # 2. Encoding
    print("Applying encoding...")
    
    # Example: You can adapt this list depending on your actual features
    label_encode_cols = ['age_group_intake', 'breed_type', 'color_group', 'intake_condition_group', 'animal_type', 'month_of_outcome']
    
    df_encoded, label_encoders = label_encode_columns(df, label_encode_cols)
    # Optional: one-hot encode if needed
    # one_hot_cols = ['animal_type', 'color_group']
    # df_encoded, onehot_encoder = one_hot_encode_columns(df_encoded, one_hot_cols)

    # 3. Causal Inference
    print("Running causal inference analysis...")
    treatment = "is_fixed"
    outcome = "outcome_group"
    common_causes = ["age_group_intake", "breed_type", "color_group"]
    causal_estimate = causal_inference_pipeline(df_encoded, treatment, outcome, common_causes)

    # 4. Define Features and Target
    print("Preparing train/test data...")
    X = df_encoded.drop(columns=[outcome])
    y = df_encoded[outcome]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Apply SMOTE-Tomek ONLY for supervised learning
    print("Applying SMOTE-Tomek on training data for modeling...")
    X_train_bal, y_train_bal = apply_smote(X_train_full, y_train_full)

    # 6. Hyperparameter Tuning (Optional: skipped for now)

    # 7. Train Random Forest
    print("Training Random Forest model...")
    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    rf_model = train_random_forest(X_train_bal, y_train_bal, best_params)

    # 8. Stacking
    print("Training stacking model...")
    base_models = [("rf", rf_model)]
    meta_model = LogisticRegression()
    stacking_model = train_stacking(meta_model, base_models, X_train_bal, y_train_bal)

    # 9. Evaluate Stacking
    print("Evaluating stacking model...")
    stacking_accuracy = evaluate_stacking(stacking_model, X_test, y_test)
    print(f"Stacking Model Accuracy: {stacking_accuracy:.4f}")

    # 10. Clustering (on raw full train set - no SMOTE)
    print("Running K-Prototypes clustering...")
    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "color_group", "breed_type"]
    clusters, kproto_model = run_kprototypes(X_train_full, numerical_features, categorical_features)

    # 11. Optional: Visualizations
    print("Plotting feature importances and SHAP...")
    # plot_feature_importances(rf_model, X_train_bal)
    # shap_summary_plot(rf_model, X_train_bal)

    print("\nFULL PIPELINE SUCCESSFULLY COMPLETED.")

if __name__ == "__main__":
    main()
