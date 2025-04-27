from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib

# Train a Stacking Classifier
def train_stacking(meta_model, base_models, X_train, y_train, cv=5):
    """
    Train a stacking classifier.

    Args:
        meta_model (object): Meta-learner (e.g., LogisticRegression, SVC).
        base_models (list): List of tuples (name, model) for base models.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cv (int): Number of cross-validation folds.

    Returns:
        stacking_clf (object): Trained StackingClassifier.
    """
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1,
        stack_method='predict_proba'
    )

    # Fit the stacking classifier
    stacking_clf.fit(X_train, y_train)
    return stacking_clf

# Evaluate the Stacking Classifier
def evaluate_stacking(stacking_clf, X_test, y_test):
    """
    Evaluate a stacking classifier.

    Args:
        stacking_clf (object): Trained StackingClassifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        accuracy (float): Accuracy score on test set.
    """
    preds = stacking_clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# Save the Stacking Model
def save_stacking_model(stacking_clf, filename):
    """
    Save the stacking classifier to a file.

    Args:
        stacking_clf (object): Trained stacking model.
        filename (str): Filename to save the model.
    """
    joblib.dump(stacking_clf, filename)
