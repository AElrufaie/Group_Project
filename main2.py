# src/main.py

import pandas as pd
from src.causal_inference.causal_analysis import causal_inference_pipeline

def main():
    # Load your dataset from the data folder
    df1 = pd.read_csv("data/animal_df.csv")

    # Define the columns for causal analysis
    treatment = "age_days_outcome"
    outcome = "los_at_shelter"
    common_causes = [
        "animal_type",
        "breed_type",
        "intake_condition_group"
    ]

    # Run causal inference pipeline
    causal_estimate, refutation_placebo, refutation_random, refutation_subset = causal_inference_pipeline(
        df1,
        treatment,
        outcome,
        common_causes
    )

    print(f"Final Causal Effect Estimate: {causal_estimate.value}")

if __name__ == "__main__":
    main()
