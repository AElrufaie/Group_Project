def test_no_nan_in_clusters():
    df = pd.DataFrame({
        'los_at_shelter': np.random.randint(1, 30, 50),
        'age_days_outcome': np.random.randint(100, 5000, 50),
        'animal_type': np.random.choice(['Dog', 'Cat'], 50),
        'has_name': np.random.choice([0, 1], 50),
        'age_group_intake': np.random.choice(['Puppy/Kitten', 'Adult'], 50),
        'month_of_outcome': np.random.choice(['January', 'February'], 50),
        'is_fixed': np.random.choice([0, 1], 50),
        'breed_type': np.random.choice(['Mix', 'Pure'], 50),
        'color_group': np.random.choice(['Black', 'White'], 50),
        'intake_condition_group': np.random.choice(['Healthy/Normal', 'Injured'], 50),
    })

    numerical_features = ["los_at_shelter", "age_days_outcome"]
    categorical_features = ["animal_type", "has_name", "age_group_intake", 
                             "month_of_outcome", "is_fixed", "breed_type", 
                             "color_group", "intake_condition_group"]

    clusters, model = run_kprototypes(df, numerical_features, categorical_features, n_clusters=5)

    assert not any(pd.isnull(clusters))
