from src.clustering.kprototypes_clustering import prepare_clustering_data, run_kprototypes
import pandas as pd

def test_clustering_output():
    """Test that clustering returns labels matching the number of samples."""
    df = pd.DataFrame({
        'age_days_outcome': [100, 200, 300],
        'los_at_shelter': [10, 20, 30],
        'animal_type': ['Dog', 'Cat', 'Dog'],
        'is_fixed': ['Yes', 'No', 'Yes'],
    })
    
    df_sample = prepare_clustering_data(df, target_column=None, sample_size=3)
    numerical_features = ['age_days_outcome', 'los_at_shelter']
    categorical_features = ['animal_type', 'is_fixed']
    
    clusters, model = run_kprototypes(df_sample, numerical_features, categorical_features, n_clusters=2)
    
    assert len(clusters) == df_sample.shape[0]
