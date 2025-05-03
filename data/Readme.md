# Dataset Overview

This folder contains raw, intermediate, and final versions of the animal dataset used throughout the ML pipeline. Each file plays a specific role during preprocessing, modeling, or evaluation.

---

## File Descriptions

| File Name                      | Description | Pipeline Stage |
|-------------------------------|-------------|----------------|
| `intake_raw.xlsx`             | Original intake data directly downloaded from the Austin Animal Center. Contains details like intake type, age, condition, etc. | Raw input |
| `outcome_raw.xlsx`            | Original outcome data (e.g., adoption, return, euthanasia) from the same source. | Raw input |
| `merged_animal_data_raw.xlsx` | Concatenated version of `intake_raw` and `outcome_raw`, joined using unique animal identifiers. May still contain missing values or inconsistent formatting. | Initial data preparation |
| `animal_df.xlsx`              | Same as above, or possibly one step earlier — used for generating SHAP or visualization (depends on script context). | Final/near-final before modeling |
| `animal_df_with_clusters.xlsx`| Adds a `kprototypes_cluster` column to `animal_df`, showing cluster assignments for each animal. Used for downstream segmentation or insight analysis. | Post-modeling (clustering) |

---

## Usage Guide

| Task | Files Used |
|------|------------|
| Initial preprocessing | `intake_raw`, `outcome_raw` |
| Feature generation & cleaning | `merged_animal_data_raw` → `animal_df_clean` |
| Model training | `animal_df_clean`, `animal_df` |
| Clustering analysis | `animal_df_with_clusters` |
| Explainability / SHAP | Likely uses `animal_df` or `animal_df_clean` |

---

## Notes

- All files are in Excel format (`.xlsx`) for easy readability and data inspection.
- Keep raw files unchanged for reproducibility.
- Processed files are regenerated each time you run `main.py`.

---

