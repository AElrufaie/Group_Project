# src/modeling/stacking.py

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib

# Import your MLflow manager
from src.mlflow_management import mlflow_manager

# Train a Stacking Classifier
def train_stacking(meta_model, base_models, X_train, y_train, cv=5):
    """
    Train a stacking classifier and track it with MLflow.

    Args:
        meta_model (object): Meta-learner (e.g., LogisticRegression, SVC).
        base_models (list): List of tuples (name, model) for base models.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cv (int): Number of cross-validation folds.

    Returns:
        stacking_clf (object): Trained StackingClassifier.
    """
    # Start MLflow run
    mlflow_manager.start_run(run_name="Stacking_Training")

    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1,
        stack_method='predict_proba'
    )

    # Fit the stacking classifier
    stacking_clf.fit(X_train, y_train)

    # Log parameters
    base_model_names = [name for name, model in base_models]
    mlflow_manager.log_params({
        "meta_model": type(meta_model).__name__,
        "base_models": ", ".join(base_model_names),
        "cv_folds": cv
    })

    # Save model
    mlflow_manager.log_model(stacking_clf, model_name="stacking_model")

    mlflow_manager.end_run()

    return stacking_clf

# Evaluate the Stacking Classifier
def evaluate_stacking(stacking_clf, X_test, y_test):
    """
    Evaluate a stacking classifier and optionally log metrics to MLflow.

    Args:
        stacking_clf (object): Trained StackingClassifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        accuracy (float): Accuracy score on test set.
    """
    preds = stacking_clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow_manager.log_metrics({"stacking_test_accuracy": acc})

    return acc

# Save the Stacking Model (still available, but optional with MLflow)
def save_stacking_model(stacking_clf, filename):
    """
    Save the stacking classifier to a file with joblib.
    
    (Note: now we also save models inside MLflow directly.)

    Args:
        stacking_clf (object): Trained stacking model.
        filename (str): Filename to save the model.
    """
    joblib.dump(stacking_clf, filename)
