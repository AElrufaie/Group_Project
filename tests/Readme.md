# `tests/` — Unit Test Suite

This folder contains unit and integration tests for each major pipeline component.  
Tests are structured to match the module layout under `src/`, ensuring modular and maintainable coverage.

---

## Folder Structure & Test Purpose

### `causal/`
| File | Description |
|------|-------------|
| `test_causal_estimate.py` | Validates the correctness of the estimated causal treatment effect (e.g., using EconML or DoWhy). |
| `test_causal_pipeline.py` | End-to-end test of the full causal inference pipeline — from preprocessing to effect estimation. |
| `test_model_creation.py` | Checks that the causal model (e.g., propensity score or uplift model) is built and trained as expected. |

---

### `clustering/`
| File | Description |
|------|-------------|
| `test_input_output.py` | Tests input shape and type handling for the k-prototypes clustering function. |
| `test_NAN_labels.py` | Ensures the model handles or removes missing/NaN values before clustering. |
| `test_num_of_cluster.py` | Verifies the correct number of clusters is returned by the model, especially when hyperparameterized. |

---

### `modeling/`
| File | Description |
|------|-------------|
| `test_hyperparameter_search.py` | Tests the Optuna-based tuning for Random Forest, XGBoost, LightGBM, etc., ensuring best params are found. |
| `test_stacking_models.py` | Validates the training and evaluation of stacking models, including meta-model integration and local saving. |

---

### `preprocessing/`
| File | Description |
|------|-------------|
| `test_feature_engineering.py` | Checks custom feature creation logic (e.g., has_name, intake age group) for consistency and correctness. |
| `test_final_cleaning.py` | Ensures missing values, dtypes, and final filtering steps behave as expected before modeling. |
| `test_prepare_dataframes.py` | Tests merging and aligning intake + outcome data, and confirms that required columns are present. |

---

## Running the Tests

From the root directory, run:

```bash
pytest
