import mlflow
import mlflow.sklearn
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

def run_kprototypes(df, numerical_features, categorical_features, n_clusters=5):
    """Run K-Prototypes clustering and log to MLflow."""
    
    # Convert categorical columns to string
    for col in categorical_features:
        df[col] = df[col].astype(str)
    
    matrix = df[numerical_features + categorical_features].to_numpy()  
    
    # Find categorical indices 
    cat_col_indices = [df.columns.get_loc(col) for col in categorical_features]

    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)

    with mlflow.start_run(run_name="k-prototypes-clustering"):
        clusters = kproto.fit_predict(matrix, categorical=cat_col_indices)
        
        silhouette = silhouette_score(matrix, clusters, metric='euclidean')  
        mlflow.log_metric("silhouette_score", silhouette)
        
        mlflow.log_param("n_clusters", n_clusters)
        
        import joblib
        joblib.dump(kproto, "kproto_model.pkl")
        mlflow.log_artifact("kproto_model.pkl")
        
        print("✅ Logged clustering model and metrics to MLflow.")
    
    return clusters, kproto
