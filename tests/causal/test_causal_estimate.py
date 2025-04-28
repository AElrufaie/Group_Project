from src.causal_inference.causal_analysis import create_causal_model, estimate_causal_effect

def test_estimate_causal_effect():
    df = pd.DataFrame({
        'age_days_outcome': [100, 200, 300, 400, 500],
        'los_at_shelter': [5, 10, 15, 20, 25],
        'animal_type': ['Dog', 'Cat', 'Dog', 'Cat', 'Dog'],
        'breed_type': ['Mix', 'Pure', 'Mix', 'Pure', 'Mix'],
        'intake_condition_group': ['Healthy', 'Injured', 'Healthy', 'Healthy', 'Injured']
    })

    treatment = 'age_days_outcome'
    outcome = 'los_at_shelter'
    common_causes = ['animal_type', 'breed_type', 'intake_condition_group']

    model = create_causal_model(df, treatment, outcome, common_causes)
    identified_estimand, causal_estimate = estimate_causal_effect(model)

    assert causal_estimate is not None
    assert isinstance(causal_estimate.value, (float, int))
