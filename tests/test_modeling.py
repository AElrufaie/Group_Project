from src.modeling.model_training import train_model
import pandas as pd
import numpy as np

def test_model_training():
    """Test that model training returns a model with a predict method."""
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [3, 2, 1]})
    y_train = np.array([0, 1, 0])
    X_val = pd.DataFrame({'feature1': [2, 1], 'feature2': [2, 3]})
    y_val = np.array([1, 0])
    
    params = {'n_estimators': 10, 'max_depth': 3}
    
    model = train_model(X_train, y_train, X_val, y_val, params)
    
    assert hasattr(model, "predict")
