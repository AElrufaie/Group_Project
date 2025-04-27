# src/mlflow_management/mlflow_manager.py

import mlflow
import mlflow.sklearn

def start_run(run_name: str = None):
    """
    Start a new MLflow run.
    
    Args:
        run_name (str, optional): Name of the run.
    """
    mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    """
    Log parameters to MLflow.
    
    Args:
        params (dict): Dictionary of parameter names and values.
    """
    mlflow.log_params(params)

def log_metrics(metrics: dict):
    """
    Log metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    mlflow.log_metrics(metrics)

def log_model(model, model_name: str = "model"):
    """
    Log a trained model to MLflow.
    
    Args:
        model: Trained model object (e.g., sklearn model).
        model_name (str, optional): Name of the model artifact.
    """
    mlflow.sklearn.log_model(model, artifact_path=model_name)

def log_artifact(file_path: str, artifact_subdir: str = None):
    """
    Log a local file or directory as an MLflow artifact.
    
    Args:
        file_path (str): Path to the file to log.
        artifact_subdir (str, optional): Subdirectory inside MLflow artifacts.
    """
    mlflow.log_artifact(local_path=file_path, artifact_path=artifact_subdir)

def end_run():
    """
    End the active MLflow run.
    """
    mlflow.end_run()
