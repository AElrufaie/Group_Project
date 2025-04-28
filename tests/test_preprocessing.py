import pandas as pd
from src.preprocessing.preprocessing import preprocess_data

def test_preprocessing_output():
    """Test that preprocessing returns a DataFrame with the same number of rows."""
    df = pd.DataFrame({
        'age_days_outcome': [10, 20, 30],
        'animal_type': ['Dog', 'Cat', 'Dog'],
        'los_at_shelter': [5, 7, 10]
    })
    
    df_cleaned = preprocess_data(df)
    
    assert isinstance(df_cleaned, pd.DataFrame)
    assert df_cleaned.shape[0] == df.shape[0]
