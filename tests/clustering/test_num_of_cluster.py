def test_unique_clusters_reasonable():
    df = pd.DataFrame({
        'los_at_shelter': np.random.randint(1, 30, 80),
        'age_days_outcome': np.random.randint(100, 5000, 80),
        'animal_type': np.random.choice(['Dog', 'Cat'], 80),
        'has_name': np.random.choice([0, 1], 80),
        'age_group_intake': np.random.choice(['Puppy/Kitten', 'Adult'], 80),
        'month_of_outcome': np.random.choice(['January', 'February'], 80),
        'is_fixed': np.random.choice([0, 1], 80),
        'breed_type': np.random.choice(['Mix', 'Pure'], 80),
        'color_group': np.random.choice(['Black', 'White'], 80),
        'intake_condition_group': np.random.choice(['Healthy/Normal', 'Injured'], 80),
    })

    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "has_name", "age_group_intake", 
                             "month_of_outcome", "is_fixed", "breed_type", 
                             "color_group", "intake_condition_group"]

    n_clusters = 5
    clusters, model = run_kprototypes(df, numerical_features, categorical_features, n_clusters=n_clusters)

    unique_clusters = len(np.unique(clusters))
    assert unique_clusters <= n_clusters
    assert unique_clusters >= 4  # at least some meaningful splits
