# src/clustering/kprototypes_clustering.py

from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import joblib
import os
import mlflow


# Import your MLflow manager
from src.mlflow_management import mlflow_manager

# --- K-Prototypes Function ---
def run_kprototypes(df, numerical_features, categorical_features, n_clusters=5):
    """Run K-Prototypes clustering and log everything to MLflow."""

    # 0. End any active MLflow run
    if mlflow.active_run() is not None:
        print("⚠️ Found an active MLflow run. Ending it first.")
        mlflow.end_run()

    # 1. Start a new run
    mlflow_manager.start_run(run_name="k-prototypes-clustering")

    # 2. Convert categorical columns to string
    for col in categorical_features:
        df[col] = df[col].astype(str)

    # 3. Prepare matrices
    matrix_all = df[numerical_features + categorical_features].to_numpy()
    matrix_numeric = df[numerical_features].to_numpy()

    # 4. Find categorical indices
    cat_col_indices = list(range(len(numerical_features), len(numerical_features) + len(categorical_features)))

    # 5. Initialize K-Prototypes
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)

    # 6. Fit model
    clusters = kproto.fit_predict(matrix_all, categorical=cat_col_indices)

    # 7. Evaluate clustering
    try:
        silhouette = silhouette_score(matrix_numeric, clusters, metric='euclidean')
        print(f"✅ Silhouette Score: {silhouette:.4f}")
    except Exception as e:
        print(f"⚠️ Silhouette score calculation failed: {e}")
        silhouette = -1

    # 8. Log parameters and metrics to MLflow
    mlflow_manager.log_params({
        "n_clusters": n_clusters,
        "init": "Cao",
        "categorical_columns": ", ".join(categorical_features)
    })
    mlflow_manager.log_metrics({
        "silhouette_score": silhouette
    })

    # 9. Save model locally
    model_filename = "kproto_model.pkl"
    joblib.dump(kproto, model_filename)

    # 10. Save clustered data locally
    clustered_df = df.copy()
    clustered_df['kprototypes_cluster'] = clusters
    clustered_filename = "animal_df_with_clusters.csv"
    clustered_df.to_csv(clustered_filename, index=False)

    # 11. Log model and data artifacts to MLflow
    mlflow_manager.log_artifact(model_filename, artifact_subdir="kproto_model")
    mlflow_manager.log_artifact(clustered_filename, artifact_subdir="clustered_data")

    # 12. End MLflow run
    mlflow_manager.end_run()

    # 13. Clean up temp files
    for file in [model_filename, clustered_filename]:
        if os.path.exists(file):
            os.remove(file)

    print("✅ K-Prototypes model and clustered data successfully logged to MLflow.")

    return clusters, kproto

    
# --- Optional Sampling Function ---
def prepare_clustering_data(df, target_column=None, sample_size=None, random_state=42):
    """Prepare data for clustering by optionally dropping target and sampling."""
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    if sample_size is not None and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=random_state)
        return df_sample
    else:
        return df  # Return full dataset if no sampling requested

