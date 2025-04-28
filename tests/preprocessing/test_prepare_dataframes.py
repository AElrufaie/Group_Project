from src.preprocessing.preprocessing import prepare_dataframes

def test_prepare_dataframes():
    intake_df = pd.DataFrame({
        'animal_id': ['a1', 'a2'],
        'datetime': ['2022-01-01T10:00:00', '2022-01-02T11:00:00'],
        'name': ['Doggo', 'Kitty'],
        'found_location': ['Austin', 'Austin'],
        'intake_type': ['Stray', 'Stray'],
        'intake_condition': ['Normal', 'Normal'],
        'animal_type': ['Dog', 'Cat'],
        'sex_upon_intake': ['Neutered Male', 'Spayed Female'],
        'age_upon_intake': ['2 years', '1 year'],
        'breed': ['Labrador Mix', 'Siamese'],
        'color': ['Black', 'White']
    })

    outcome_df = pd.DataFrame({
        'animal_id': ['a1', 'a2'],
        'datetime': ['2022-01-10T10:00:00', '2022-01-12T11:00:00'],
        'monthyear': ['2022-01', '2022-01'],
        'date_of_birth': ['2020-01-01', '2021-01-01'],
        'outcome_type': ['Adoption', 'Adoption'],
        'outcome_subtype': ['Foster', 'Foster'],
        'animal_type': ['Dog', 'Cat'],
        'sex_upon_outcome': ['Neutered Male', 'Spayed Female'],
        'age_upon_outcome': ['2 years', '1 year'],
        'breed': ['Labrador Mix', 'Siamese'],
        'color': ['Black', 'White']
    })

    animal_df = prepare_dataframes(intake_df, outcome_df)
    expected_cols = ['name_intake', 'name_outcome', 'datetime_intake', 'datetime_outcome']
    
    for col in expected_cols:
        assert col in animal_df.columns
    assert len(animal_df) == 2
