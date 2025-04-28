from src.causal_inference.causal_analysis import causal_inference_pipeline

def test_causal_inference_pipeline():
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

    causal_estimate, refutation_placebo, refutation_random, refutation_subset = causal_inference_pipeline(
        df, treatment, outcome, common_causes
    )

    assert causal_estimate is not None
    assert isinstance(causal_estimate.value, (float, int))
    assert hasattr(refutation_placebo, 'estimated_effect')
    assert hasattr(refutation_random, 'estimated_effect')
    assert hasattr(refutation_subset, 'estimated_effect')
