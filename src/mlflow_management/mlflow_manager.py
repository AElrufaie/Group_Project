# src/mlflow_management/mlflow_manager.py
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

def start_run(run_name: str = None):
    """Start a new MLflow run."""
    mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    """Log parameters to MLflow."""
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    """Log metrics to MLflow."""
    mlflow.log_metrics(metrics)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def log_model(model, model_name: str = "model", X_train=None, y_train=None):
    """
    Log a trained model to MLflow with optional input signature and input example.
    
    Args:
        model: Trained model object (e.g., sklearn model).
        model_name (str, optional): Name of the model artifact.
        X_train: Optional training features for signature and example.
        y_train: Optional training labels for signature.
    """
    if X_train is not None and y_train is not None:
        X_train_safe = X_train.copy()
        
        # --- Convert all integer columns to float64
        for col in X_train_safe.select_dtypes(include=['int']).columns:
            X_train_safe[col] = X_train_safe[col].astype('float64')


        signature = infer_signature(X_train_safe, y_train)
        input_example = X_train_safe.sample(1).to_dict(orient="records")[0]

        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example
        )
    else:
        mlflow.sklearn.log_model(model, artifact_path=model_name)

def log_artifact(local_path: str, artifact_subdir: str = None):
    """
    Log a local file or directory as an MLflow artifact.

    Args:
        local_path (str): Path to the file or directory to log.
        artifact_subdir (str, optional): Subdirectory within the artifacts.
    """
    if artifact_subdir:
        mlflow.log_artifact(local_path, artifact_path=artifact_subdir)
    else:
        mlflow.log_artifact(local_path)


def end_run():
    """End the active MLflow run."""
    mlflow.end_run()
