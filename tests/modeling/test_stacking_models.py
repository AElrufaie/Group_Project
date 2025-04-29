import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.datasets import make_classification
from src.modeling.stacking import (
    get_meta_models,
    get_base_model_combinations,
    train_and_evaluate_stacking_models
)

@pytest.fixture
def small_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def dummy_best_params():
    return {
        'xgb': {'n_estimators': 10, 'max_depth': 3},
        'rf': {'n_estimators': 10, 'max_depth': 3},
        'cat': {'iterations': 10, 'depth': 3}
    }

def test_get_meta_models():
    meta_models = get_meta_models()
    assert isinstance(meta_models, dict)
    assert "Logistic Regression" in meta_models
    assert "Linear SVM" in meta_models
    assert "Decision Tree" in meta_models

def test_get_base_model_combinations(dummy_best_params):
    base_model_combinations = get_base_model_combinations(dummy_best_params)
    assert isinstance(base_model_combinations, list)
    assert isinstance(base_model_combinations[0], list)
    assert len(base_model_combinations[0]) == 3

@patch("src.modeling.stacking.mf_manager.start_run")
@patch("src.modeling.stacking.mf_manager.log_params")
@patch("src.modeling.stacking.mf_manager.log_metrics")
@patch("src.modeling.stacking.mf_manager.log_model")
@patch("src.modeling.stacking.mf_manager.end_run")
def test_train_and_evaluate_stacking_models(mock_end_run, mock_log_model, mock_log_metrics, mock_log_params, mock_start_run, small_data, dummy_best_params):
    X, y = small_data
    meta_models = get_meta_models()
    base_model_combinations = get_base_model_combinations(dummy_best_params)

    # Only take one meta model to keep test fast
    small_meta_models = {k: v for i, (k, v) in enumerate(meta_models.items()) if i == 0}

    train_and_evaluate_stacking_models(
        small_meta_models,
        base_model_combinations,
        X,
        y,
        X,
        y
    )

    # Verify that mlflow functions were called
    assert mock_start_run.called
    assert mock_log_params.called
    assert mock_log_metrics.called
    assert mock_log_model.called
    assert mock_end_run.called
