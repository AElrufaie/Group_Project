import pandas as pd
from src.preprocessing.preprocessing import final_cleaning

def test_final_cleaning():
    animal_df = pd.DataFrame({
        'animal_id': ['a1'],
        'datetime_intake': [pd.Timestamp('2022-01-01')],
        'datetime_outcome': [pd.Timestamp('2022-01-10')],
        'age_days_intake': [500],
        'los_at_shelter': [9],
        'outcome_group': ['Positive']
    })

    cleaned_df = final_cleaning(animal_df)

    assert 'animal_id' not in cleaned_df.columns
    assert 'datetime_intake' not in cleaned_df.columns
    assert 'age_days_intake' not in cleaned_df.columns
    assert 'los_at_shelter' in cleaned_df.columns
    assert 'outcome_group' in cleaned_df.columns
