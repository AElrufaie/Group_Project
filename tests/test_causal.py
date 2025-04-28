from src.causal_inference.causal_analysis import causal_inference_pipeline
import pandas as pd

def test_causal_pipeline_runs():
    """Test that causal pipeline returns a non-null causal effect."""
    df = pd.DataFrame({
        'age_days_outcome': [100, 200, 300],
        'los_at_shelter': [10, 20, 30],
        'animal_type': ['Dog', 'Cat', 'Dog'],
        'breed_type': ['Mixed', 'Pure', 'Mixed'],
        'intake_condition_group': ['Healthy', 'Injured', 'Healthy']
    })
    
    treatment = 'age_days_outcome'
    outcome = 'los_at_shelter'
    common_causes = ['animal_type', 'breed_type', 'intake_condition_group']
    
    causal_estimate, _, _, _ = causal_inference_pipeline(df, treatment, outcome, common_causes)
    
    assert causal_estimate.value is not None
