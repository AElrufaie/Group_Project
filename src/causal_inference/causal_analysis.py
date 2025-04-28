"""
Causal Inference module using DoWhy.
"""

import pandas as pd
import dowhy
from dowhy import CausalModel
import mlflow

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    return identified_estimand, causal_estimate

def causal_inference_pipeline(df, treatment, outcome, common_causes):
    """Run the full causal inference pipeline and log to MLflow."""
    
    with mlflow.start_run(run_name="causal-inference-dowhy"):
        model = create_causal_model(df, treatment, outcome, common_causes)
        
        identified_estimand, causal_estimate = estimate_causal_effect(model)
        
        # Refutations
        refutation_placebo = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="placebo_treatment_refuter"
        )
        
        refutation_random = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="random_common_cause"
        )
        
        refutation_subset = model.refute_estimate(
            identified_estimand,
            causal_estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.9
        )
        
        # Log parameters
        mlflow.log_param("treatment", treatment)
        mlflow.log_param("outcome", outcome)
        mlflow.log_param("method", "backdoor.linear_regression")
        
        # Log causal effect
        mlflow.log_metric("causal_effect", causal_estimate.value)
        
        # Log refutation results
        mlflow.log_metric("placebo_refutation_effect", refutation_placebo.estimated_effect)
        mlflow.log_metric("random_common_cause_refutation_effect", refutation_random.estimated_effect)
        mlflow.log_metric("data_subset_refutation_effect", refutation_subset.estimated_effect)

        print(f"✅ Estimated Causal Effect: {causal_estimate.value}")
        print(f"✅ Refutation (Placebo): {refutation_placebo}")
        print(f"✅ Refutation (Random Common Cause): {refutation_random}")
        print(f"✅ Refutation (Data Subset): {refutation_subset}")
    
    return causal_estimate, refutation_placebo, refutation_random, refutation_subset
