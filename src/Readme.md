# `src/` — Source Code Directory

This folder contains the full source code for our end-to-end machine learning pipeline, including data processing, training, evaluation, explainability, and model serving.

---

## 🧭 Folder Overview

| Folder / File         | Description |
|------------------------|-------------|
| `main.py`              | Main entry point to run the full pipeline (preprocessing → modeling → evaluation). |
| `causal_inference/`    | Implements causal inference logic to estimate treatment effects (e.g., do fixed animals stay longer?). |
| `clustering/`          | Contains k-prototypes and other clustering logic to group similar animal profiles. |
| `data_drafting/`       | Preprocessing helpers for data loading, cleaning, and feature extraction. |
| `mlflow_management/`   | Centralized MLflow tracking module — handles logging of parameters, metrics, and models. |
| `modeling/`            | Core ML code: stacking models, hyperparameter tuning, SMOTE balancing, and evaluation. |
| `preprocessing/`       | Feature engineering, label encoding, and final dataset preparation for training. |
| `saved_models/`        | Stores serialized `.pkl` model files for local reuse or deployment. |
| `SHAP_value/`          | Model explainability using SHAP: plots and scripts to visualize feature impacts. |

---

## 🚀 How to Run the Pipeline

Before running the main function, start the MLflow server in a separate terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 127.0.0.1 --port 5000

