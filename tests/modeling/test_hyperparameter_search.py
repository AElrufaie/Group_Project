import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from unittest.mock import MagicMock

from src.modeling.hyperparameter_search import (
    tune_random_forest,
    tune_xgboost,
    tune_lightgbm,
    tune_catboost,
    tune_decision_tree
)

@pytest.fixture
def small_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def mock_trial():
    trial = MagicMock()
    trial.suggest_int.side_effect = lambda name, low, high: low
    trial.suggest_float.side_effect = lambda name, low, high: low
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    return trial

def test_tune_random_forest(small_data, mock_trial):
    X, y = small_data
    acc = tune_random_forest(mock_trial, X, X, y, y)
    assert 0.0 <= acc <= 1.0

def test_tune_xgboost(small_data, mock_trial):
    X, y = small_data
    acc = tune_xgboost(mock_trial, X, X, y, y)
    assert 0.0 <= acc <= 1.0

def test_tune_lightgbm(small_data, mock_trial):
    X, y = small_data
    acc = tune_lightgbm(mock_trial, X, X, y, y)
    assert 0.0 <= acc <= 1.0

def test_tune_catboost(small_data, mock_trial):
    X, y = small_data
    acc = tune_catboost(mock_trial, X, X, y, y)
    assert 0.0 <= acc <= 1.0

def test_tune_decision_tree(small_data, mock_trial):
    X, y = small_data
    acc = tune_decision_tree(mock_trial, X, X, y, y)
    assert 0.0 <= acc <= 1.0
