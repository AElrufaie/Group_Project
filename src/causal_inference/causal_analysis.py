"""
Causal Inference module using DoWhy.
"""

import pandas as pd
import dowhy
from dowhy import CausalModel
import mlflow

def create_causal_model(df, treatment, outcome, common_causes):
    """Create and return a CausalModel object."""
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes
    )
    return model

def estimate_causal_effect(model, method_name="backdoor.linear_regression"):
    """Identify and estimate causal effect."""
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand, method_name=method_name)
    return causal_estimate

def causal_inference_pipeline(df, treatment, outcome, common_causes):
    """Run the full causal inference pipeline and log to MLflow."""
    
    with mlflow.start_run(run_name="causal-inference-dowhy"):
        model = create_causal_model(df, treatment, outcome, common_causes)
        
        causal_estimate = estimate_causal_effect(model)
        
        # Log parameters and result
        mlflow.log_param("treatment", treatment)
        mlflow.log_param("outcome", outcome)
        mlflow.log_param("method", "backdoor.linear_regression")
        
        mlflow.log_metric("causal_effect", causal_estimate.value)
        
        print(f"Estimated Causal Effect: {causal_estimate.value}")
    
    return causal_estimate
