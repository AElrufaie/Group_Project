# src/clustering/kprototypes_clustering.py

from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import joblib
import os

# Import your MLflow manager
from src.mlflow_management import mlflow_manager

def run_kprototypes(df, numerical_features, categorical_features, n_clusters=5):
    """Run K-Prototypes clustering and log everything to MLflow."""

    # Convert categorical columns to string
    for col in categorical_features:
        df[col] = df[col].astype(str)
    
    # Prepare matrix
    matrix_all = df[numerical_features + categorical_features].to_numpy()
    matrix_numeric = df[numerical_features].to_numpy()
    
    # Find categorical indices
    cat_col_indices = [df.columns.get_loc(col) for col in categorical_features]

    # Initialize model
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)

    # Start MLflow tracking
    mlflow_manager.start_run(run_name="k-prototypes-clustering")

    # Fit model
    clusters = kproto.fit_predict(matrix_all, categorical=[df.columns.get_loc(col) for col in categorical_features])

    # Evaluate clustering
    silhouette = silhouette_score(matrix_numeric, clusters, metric='euclidean')
    
    # Log params and metrics
    mlflow_manager.log_params({
        "n_clusters": n_clusters,
        "init": "Cao",
        "categorical_columns": ", ".join(categorical_features)
    })
    mlflow_manager.log_metrics({
        "silhouette_score": silhouette
    })

    # Save model locally
    model_filename = "kproto_model.pkl"
    joblib.dump(kproto, model_filename)

    # Log model artifact
    mlflow_manager.log_artifact(model_filename, artifact_subdir="kproto_model")

    # End MLflow run
    mlflow_manager.end_run()

    # Optional: remove temp file to stay clean
    if os.path.exists(model_filename):
        os.remove(model_filename)

    print("K-Prototypes model and metrics logged successfully to MLflow.")

    return clusters, kproto

def prepare_clustering_data(df, target_column=None, sample_size=30000, random_state=42):
    """Prepare data for clustering by dropping target and sampling."""
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    df_sample = df.sample(n=sample_size, random_state=random_state)
    return df_sample

