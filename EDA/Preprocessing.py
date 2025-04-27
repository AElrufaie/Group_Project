import pandas as pd
import numpy as np
import requests
import time
from io import StringIO

# --- DATA DOWNLOAD FUNCTIONS ---

def get_data(base_url, offset, limit=1000):
    """Fetch paginated data from the API."""
    params = {
        "$limit": limit,
        "$offset": offset
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        return pd.DataFrame()

def download_data(base_url_1, base_url_2, limit=1000):
    """Download and concatenate paginated data from two sources."""
    offset = 0
    all_data_1, all_data_2 = [], []

    while True:
        df1 = get_data(base_url_1, offset, limit)
        df2 = get_data(base_url_2, offset, limit)

        if df1.empty and df2.empty:
            break

        all_data_1.append(df1)
        all_data_2.append(df2)
        offset += limit
        time.sleep(1)  # Avoid API rate limiting

    final_df1 = pd.concat(all_data_1, ignore_index=True)
    final_df2 = pd.concat(all_data_2, ignore_index=True)
    return final_df1, final_df2

# --- DATA CLEANING FUNCTIONS ---

def merge_latest_intake_outcome(df_intake, df_outcome):
    """Merge latest intake and outcome records for each animal."""
    df_intake['datetime'] = pd.to_datetime(df_intake['datetime'], errors='coerce')
    df_outcome['datetime'] = pd.to_datetime(df_outcome['datetime'], errors='coerce')

    latest_intake = df_intake.sort_values(by='datetime', ascending=False).drop_duplicates('animal_id')
    final_outcome = df_outcome.sort_values(by='datetime', ascending=False).drop_duplicates('animal_id')

    # Rename columns
    latest_intake = latest_intake.rename(columns={
        'name': 'name_intake', 'datetime': 'datetime_intake', 'found_location': 'found_location',
        'intake_type': 'intake_type', 'intake_condition': 'intake_condition', 'animal_type': 'animal_type_intake',
        'sex_upon_intake': 'sex_upon_intake', 'age_upon_intake': 'age_upon_intake',
        'breed': 'breed_intake', 'color': 'color_intake'
    })

    final_outcome = final_outcome.rename(columns={
        'name': 'name_outcome', 'datetime': 'datetime_outcome', 'monthyear': 'monthyear_outcome',
        'date_of_birth': 'date_of_birth_outcome', 'outcome_type': 'outcome_type',
        'outcome_subtype': 'outcome_subtype', 'animal_type': 'animal_type_outcome',
        'sex_upon_outcome': 'sex_upon_outcome', 'age_upon_outcome': 'age_upon_outcome',
        'breed': 'breed_outcome', 'color': 'color_outcome'
    })

    # Merge
    animal_df = pd.merge(latest_intake, final_outcome, on='animal_id', how='inner')
    return animal_df

def drop_unnecessary_columns(df):
    """Drop early unnecessary columns."""
    df.drop('outcome_subtype', axis=1, inplace=True)
    df['has_name'] = df['name_intake'].notna().astype(int)
    df.drop(['name_intake', 'name_outcome'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

def create_features(df):
    """Create age, LOS, month, outcome group."""
    df['outcome_group'] = df['outcome_type'].apply(group_outcome)
    df = df[df['outcome_group'] != 'Other']

    for col in ['datetime_intake', 'datetime_outcome', 'date_of_birth_outcome']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['age_days_intake'] = (df['datetime_intake'] - df['date_of_birth_outcome']).dt.days
    df['age_days_outcome'] = (df['datetime_outcome'] - df['date_of_birth_outcome']).dt.days

    df = df[(df['age_days_intake'] > 0) & (df['age_days_outcome'] > 0)]

    df['age_group_intake'] = df['age_days_intake'].apply(classify_age)
    df['age_group_outcome'] = df['age_days_outcome'].apply(classify_age)

    df['los_at_shelter'] = (df['datetime_outcome'] - df['datetime_intake']).dt.days
    df = df[df['los_at_shelter'] > 0]

    df['month_of_outcome'] = df['datetime_outcome'].dt.strftime('%B')
    return df

def simplify_features(df):
    """Simplify and group some categorical features."""
    df['reproductive_status'] = df['sex_upon_outcome'].apply(simplify_sex)
    df['is_fixed'] = df['reproductive_status'].apply(lambda x: 1 if x == 'Fixed' else 0)
    df['breed_type'] = df['breed_intake'].apply(lambda x: 'Mix' if 'Mix' in x else 'Pure')
    df['color_group'] = df['color_intake'].apply(simplify_color)
    df['intake_condition_group'] = df['intake_condition'].apply(group_intake_condition)
    df.rename(columns={'animal_type_intake': 'animal_type'}, inplace=True)
    if 'animal_type_outcome' in df.columns:
        df.drop(columns=['animal_type_outcome'], inplace=True)
    return df

def final_cleanup(df):
    """Final drop of unneeded columns."""
    final_drop_cols = [
        'animal_id', 'datetime_intake', 'datetime_outcome', 'datetime2', 'found_location',
        'intake_type', 'intake_condition', 'breed_intake', 'color_intake', 'monthyear_outcome',
        'date_of_birth_outcome', 'outcome_type', 'sex_upon_intake', 'sex_upon_outcome',
        'age_upon_intake', 'age_upon_outcome', 'breed_outcome', 'color_outcome',
        'reproductive_status', 'age_group_outcome', 'age_days_intake'
    ]
    df.drop(columns=[col for col in final_drop_cols if col in df.columns], inplace=True)
    return df

def clean_animal_data(df):
    """Full cleaning pipeline: apply modular steps."""
    df = drop_unnecessary_columns(df)
    df = create_features(df)
    df = simplify_features(df)
    df = final_cleanup(df)
    return df

# --- HELPER FUNCTIONS ---

def group_outcome(outcome):
    if outcome in ['Adoption', 'Return to Owner', 'Rto-Adopt']:
        return 'Positive'
    elif outcome in ['Transfer', 'Relocate']:
        return 'Neutral'
    elif outcome in ['Euthanasia', 'Died', 'Disposal', 'Lost', 'Missing', 'Stolen']:
        return 'Negative'
    else:
        return 'Other'

def classify_age(days):
    if days < 365:
        return 'Puppy/Kitten'
    elif days < 1095:
        return 'Young Adult'
    elif days < 2555:
        return 'Adult'
    else:
        return 'Senior'

def simplify_sex(sex):
    if sex in ['Neutered Male', 'Spayed Female']:
        return 'Fixed'
    elif sex in ['Intact Male', 'Intact Female']:
        return 'Intact'
    else:
        return 'Unknown'

def simplify_color(color):
    color = str(color).lower()
    if any(word in color for word in ['black', 'ebony', 'charcoal']):
        return 'Black'
    elif any(word in color for word in ['white', 'cream', 'ivory', 'pearl', 'platinum']):
        return 'White'
    elif any(word in color for word in ['brown', 'brindle', 'chocolate', 'mocha', 'mahogany']):
        return 'Brown'
    elif any(word in color for word in ['gray', 'grey', 'blue', 'silver', 'slate', 'pewter', 'ash']):
        return 'Grey'
    elif any(word in color for word in ['gold', 'tan', 'yellow', 'fawn', 'buff', 'sandy']):
        return 'Gold'
    elif any(word in color for word in ['red', 'ginger', 'orange', 'copper', 'auburn']):
        return 'Red'
    elif any(word in color for word in ['cream', 'beige', 'off-white']):
        return 'Cream'
    elif 'tabby' in color:
        return 'Tabby'
    elif 'merle' in color:
        return 'Merle'
    elif 'tricolor' in color or ('black' in color and 'tan' in color and 'white' in color):
        return 'Tricolor'
    elif any(word in color for word in ['spotted', 'speckled', 'ticked', 'roan', 'freckled']):
        return 'Spotted'
    elif '/' in color or 'and' in color:
        return 'Multicolor'
    else:
        return 'Other'

def group_intake_condition(condition):
    condition = str(condition).lower()
    if condition in ['normal']:
        return 'Healthy/Normal'
    elif condition in ['injured', 'neurologic', 'med attn', 'med urgent']:
        return 'Injured'
    elif condition in ['sick', 'medical', 'parvo', 'panleuk', 'congenital']:
        return 'Sick/Medical'
    elif condition in ['behavior', 'feral']:
        return 'Behavioral Issues'
    elif condition in ['neonatal', 'nursing', 'aged', 'pregnant']:
        return 'Life Stage/Developmental'
    else:
        return 'Other/Unknown'


