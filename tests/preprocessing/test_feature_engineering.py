import pandas as pd
from src.preprocessing.preprocessing import feature_engineering

def test_feature_engineering():
    animal_df = pd.DataFrame({
        'animal_id': ['a1'],
        'datetime_intake': pd.to_datetime(['2022-01-01T10:00:00']),
        'datetime_outcome': pd.to_datetime(['2022-05-10T10:00:00']),
        'name_intake': ['Doggo'],
        'outcome_type': ['Adoption'],
        'animal_type_outcome': ['Dog'],
        'sex_upon_outcome': ['Neutered Male'],
        'breed_intake': ['Labrador Mix'],
        'color_intake': ['Black'],
        'intake_condition': ['Normal'],
        'date_of_birth_outcome': pd.to_datetime(['2020-01-01'])
    })
    
    processed_df = feature_engineering(animal_df)

    assert 'los_at_shelter' in processed_df.columns
    assert processed_df['los_at_shelter'].iloc[0] == 129
    assert 'outcome_group' in processed_df.columns
    assert processed_df['outcome_group'].iloc[0] == 'Positive'
    assert 'has_name' in processed_df.columns
    assert processed_df['has_name'].iloc[0] == 1
