import pandas as pd
import numpy as np
from src.clustering.kprototypes_clustering import run_kprototypes

def test_clusters_length_matches_input():
    df = pd.DataFrame({
        'los_at_shelter': np.random.randint(1, 30, 100),
        'age_days_outcome': np.random.randint(100, 5000, 100),
        'animal_type': np.random.choice(['Dog', 'Cat'], 100),
        'has_name': np.random.choice([0, 1], 100),
        'age_group_intake': np.random.choice(['Puppy/Kitten', 'Adult'], 100),
        'month_of_outcome': np.random.choice(['January', 'February'], 100),
        'is_fixed': np.random.choice([0, 1], 100),
        'breed_type': np.random.choice(['Mix', 'Pure'], 100),
        'color_group': np.random.choice(['Black', 'White'], 100),
        'intake_condition_group': np.random.choice(['Healthy/Normal', 'Injured'], 100),
    })

    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "has_name", "age_group_intake", 
                             "month_of_outcome", "is_fixed", "breed_type", 
                             "color_group", "intake_condition_group"]

    clusters, model = run_kprototypes(df, numerical_features, categorical_features, n_clusters=5)

    assert len(clusters) == len(df)
