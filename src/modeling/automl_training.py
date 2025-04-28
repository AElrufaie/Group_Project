# --- Imports ---
import os
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Import mlflow manager (your team's module)
from src.mlflow_management.mlflow_manager import (
    start_run,
    log_params,
    log_metrics,
    log_model,
    log_artifact,
    end_run
)

# --- Set base path ---
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define dataset and model paths
csv_path = os.path.join(base_dir, "data", "animal_df.csv")
model_save_folder = os.path.join(base_dir, "models")

# --- Load dataset ---
df_pd = pd.read_csv(csv_path)

# --- Start H2O cluster ---
h2o.init()

# --- Convert to H2O Frame ---
df_h2o = h2o.H2OFrame(df_pd)

# --- Ensure target is treated as classification ---
target = 'outcome_group'
df_h2o[target] = df_h2o[target].asfactor()
features = df_h2o.columns
features.remove(target)

# --- Split into train/test ---
train, test = df_h2o.split_frame(ratios=[0.8], seed=42)

# --- Run AutoML ---
aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=600)
aml.train(x=features, y=target, training_frame=train)

# --- View Leaderboard ---
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# --- Save Best Model Locally ---
model_save_path = h2o.save_model(model=aml.leader, path=model_save_folder, force=True)
print(f"✅ Best model saved to: {model_save_path}")

# --- Evaluate Best Model on Test Set ---
perf = aml.leader.model_performance(test_data=test)
hit_ratio_table = perf.hit_ratio_table()
top1_accuracy = hit_ratio_table.cell_values[1][1]  # Top-1 accuracy (normal accuracy)

# --- Print Evaluation Metrics ---
print("\n=== Best Model Performance on Test Set ===")
print(f"Accuracy: {top1_accuracy:.4f}")
print(f"Mean Per Class Error: {perf.mean_per_class_error():.4f}")
print(f"Logloss: {perf.logloss():.4f}")
print("Confusion Matrix:")
print(perf.confusion_matrix())

# --- Track Everything via mlflow_manager ---
print("\n🚀 Logging model and metrics to MLflow via mlflow_manager...")

# Start MLflow run
start_run(run_name="H2O_AutoML_Run")

# Log parameters if needed
params = {
    "max_models": 10,
    "max_runtime_secs": 600,
    "seed": 42,
    "target_column": target
}
log_params(params)

# Log metrics
metrics = {
    "accuracy": top1_accuracy,
    "mean_per_class_error": perf.mean_per_class_error(),
    "logloss": perf.logloss()
}
log_metrics(metrics)

# Log model
log_artifact(model_save_path, artifact_subdir="h2o_automl_model")

# End MLflow run
end_run()

print("✅ MLflow logging complete using mlflow_manager!")

# --- Shutdown H2O cleanly ---
h2o.cluster().shutdown()
