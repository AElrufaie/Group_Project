from src.mlflow_management import mlflow_manager as mf_manager
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import os
import joblib

# Meta-Models
def get_meta_models():
    return {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, solver='saga', penalty='l2',
            n_jobs=-1, verbose=1, warm_start=True, multi_class='multinomial'
        ),
        'Linear SVM': LinearSVC(
            random_state=42, max_iter=1000, verbose=1, dual=False
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=None
        )
    }

# Base Model Combinations
def get_base_model_combinations(best_params_dict):
    return [
        [('xgb', XGBClassifier(**best_params_dict['xgb'], random_state=42, verbosity=1)),
         ('rf', RandomForestClassifier(**best_params_dict['rf'], random_state=42, n_jobs=-1)),
         ('catboost', CatBoostClassifier(**best_params_dict['cat'], random_seed=42, logging_level='Silent'))],
    ]

# Train and Evaluate Stacking
def train_and_evaluate_stacking_models(meta_models, base_model_combinations, X_train, y_train, X_test, y_test):
    stacking_results = {}

    # Create 'saved_models' folder if it does not exist
    saved_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    for meta_name, meta_model in meta_models.items():
        for i, base_models in enumerate(base_model_combinations):
            print(f"\nTraining Stacking Model {i+1} with Meta-Model: {meta_name}")

            # Start MLflow run
            mf_manager.start_run(run_name=f"{meta_name}_Combination_{i+1}")

            # Log meta-model and base models
            base_model_names = [name for name, _ in base_models]
            params_to_log = {
                "Meta_Model": meta_name,
                "Base_Models": str(base_model_names)
            }
            mf_manager.log_params(params_to_log)

            # Manually Train Base Models
            trained_base_models = []
            for name, model in tqdm(base_models, desc=f"Training Base Models for {meta_name}"):
                model.fit(X_train, y_train)
                trained_base_models.append((name, model))

            # Train Stacking Classifier
            stacking_clf = StackingClassifier(
                estimators=trained_base_models,
                final_estimator=meta_model,
                cv=2,
                n_jobs=1,
                stack_method='predict_proba'
            )

            stacking_clf.fit(X_train, y_train)

            # Evaluate
            stacking_preds = stacking_clf.predict(X_test)
            stacking_accuracy = accuracy_score(y_test, stacking_preds)

            stacking_results[f"{meta_name} - Combination {i+1}"] = stacking_accuracy

            print(f"\nStacking Model {i+1} with {meta_name} Accuracy: {stacking_accuracy:.4f}")

            cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=2, scoring='accuracy')
            print("Cross-Validated Accuracy Scores:", cv_scores)
            print("Mean Accuracy:", cv_scores.mean())

            # Log metrics
            metrics_to_log = {
                "Test_Accuracy": stacking_accuracy,
                "CV_Mean_Accuracy": cv_scores.mean(),
                "CV_Std_Accuracy": cv_scores.std()
            }
            mf_manager.log_metrics(metrics_to_log)

            # Log model to MLflow
            mf_manager.log_model(stacking_clf, model_name="stacking_model", X_train=X_train, y_train=y_train)

            # Save model as local .pkl file
            model_filename = f"stacking_{meta_name.replace(' ', '_')}_combination_{i+1}.pkl"
            model_save_path = os.path.join(saved_models_dir, model_filename)
            joblib.dump(stacking_clf, model_save_path, compress=3)
            print(f"✅ Model saved locally at: {model_save_path}")

            # End MLflow run
            mf_manager.end_run()

    # Print all final results
    print("\n📈 Final Stacking Model Performance Comparison:")
    for model_name, acc in stacking_results.items():
        print(f"{model_name}: {acc:.4f}")
