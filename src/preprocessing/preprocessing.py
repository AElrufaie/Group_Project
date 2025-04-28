# src/preprocessing/preprocessing.py

import pandas as pd
import numpy as np
import time
import requests
from io import StringIO
import os

# --- Fetching Data ---
def fetch_data_from_web(base_url, limit=1000):
    offset = 0
    all_data = []

    def get_data(base_url, offset):
        params = {"$limit": limit, "$offset": offset}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        else:
            return pd.DataFrame()

    while True:
        df = get_data(base_url, offset)
        if df.empty:
            break
        all_data.append(df)
        offset += limit
        time.sleep(1)

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- Robust Datetime Parsing ---
def robust_datetime_parsing(df, column_name):
    original = df[column_name].copy()
    df[column_name] = pd.to_datetime(original, utc=True, errors='coerce')
    mask_failed = df[column_name].isna()
    df.loc[mask_failed, column_name] = pd.to_datetime(
        original[mask_failed],
        format='%Y-%m-%dT%H:%M:%S',
        utc=True,
        errors='coerce'
    )

# --- Prepare DataFrames ---
def prepare_dataframes(df_intake, df_outcome):
    df_intake['animal_id'] = df_intake['animal_id'].str.strip().str.lower()
    df_outcome['animal_id'] = df_outcome['animal_id'].str.strip().str.lower()

    robust_datetime_parsing(df_intake, 'datetime')
    robust_datetime_parsing(df_outcome, 'datetime')

    latest_intake = df_intake.sort_values(by='datetime', ascending=False).drop_duplicates(subset='animal_id')
    latest_outcome = df_outcome.sort_values(by='datetime', ascending=False).drop_duplicates(subset='animal_id')

    common_animal_ids = set(latest_intake['animal_id']).intersection(set(latest_outcome['animal_id']))
    latest_intake = latest_intake[latest_intake['animal_id'].isin(common_animal_ids)]
    latest_outcome = latest_outcome[latest_outcome['animal_id'].isin(common_animal_ids)]

    latest_intake = latest_intake.rename(columns={
        'name': 'name_intake',
        'datetime': 'datetime_intake',
        'found_location': 'found_location',
        'intake_type': 'intake_type',
        'intake_condition': 'intake_condition',
        'animal_type': 'animal_type_intake',
        'sex_upon_intake': 'sex_upon_intake',
        'age_upon_intake': 'age_upon_intake',
        'breed': 'breed_intake',
        'color': 'color_intake'
    })

    latest_outcome = latest_outcome.rename(columns={
        'name': 'name_outcome',
        'datetime': 'datetime_outcome',
        'monthyear': 'monthyear_outcome',
        'date_of_birth': 'date_of_birth_outcome',
        'outcome_type': 'outcome_type',
        'outcome_subtype': 'outcome_subtype',
        'animal_type': 'animal_type_outcome',
        'sex_upon_outcome': 'sex_upon_outcome',
        'age_upon_outcome': 'age_upon_outcome',
        'breed': 'breed_outcome',
        'color': 'color_outcome'
    })

    animal_df = pd.merge(latest_intake, latest_outcome, on='animal_id', how='inner')

    if 'outcome_subtype' in animal_df.columns:
        animal_df.drop(columns=['outcome_subtype'], inplace=True)

    return animal_df

# --- Feature Engineering ---
def feature_engineering(animal_df):
    animal_df['has_name'] = animal_df['name_intake'].notna().astype(int)

    essential_cols = ['datetime_intake', 'datetime_outcome']
    dob_col = 'date_of_birth_outcome' if 'date_of_birth_outcome' in animal_df.columns else 'date_of_birth'
    essential_cols.append(dob_col)
    animal_df.dropna(subset=essential_cols, inplace=True)

    animal_df['outcome_group'] = animal_df['outcome_type'].apply(group_outcome)
    animal_df = animal_df[animal_df['outcome_group'] != 'Other'].copy()

    date_cols = ['datetime_intake', 'datetime_outcome']
    if 'date_of_birth_outcome' in animal_df.columns:
        date_cols.append('date_of_birth_outcome')
    elif 'date_of_birth' in animal_df.columns:
        date_cols.append('date_of_birth')

    for col in date_cols:
        animal_df[col] = pd.to_datetime(animal_df[col], errors='coerce')
        if animal_df[col].dt.tz is not None:
            animal_df[col] = animal_df[col].dt.tz_convert(None)

    dob_col = 'date_of_birth_outcome' if 'date_of_birth_outcome' in animal_df.columns else 'date_of_birth'

    animal_df['age_days_intake'] = (animal_df['datetime_intake'] - animal_df[dob_col]).dt.days
    animal_df['age_days_outcome'] = (animal_df['datetime_outcome'] - animal_df[dob_col]).dt.days
    animal_df = animal_df[(animal_df['age_days_intake'] > 0) & (animal_df['age_days_outcome'] > 0)].copy()

    animal_df['age_group_intake'] = animal_df['age_days_intake'].apply(classify_age)
    animal_df['age_group_outcome'] = animal_df['age_days_outcome'].apply(classify_age)

    animal_df['los_at_shelter'] = (animal_df['datetime_outcome'] - animal_df['datetime_intake']).dt.days
    animal_df = animal_df[animal_df['los_at_shelter'] > 0].copy()

    animal_df['month_of_outcome'] = animal_df['datetime_outcome'].dt.strftime('%B')
    animal_df['reproductive_status'] = animal_df['sex_upon_outcome'].apply(simplify_sex)
    animal_df['is_fixed'] = animal_df['reproductive_status'].apply(lambda x: 1 if x == 'Fixed' else 0)

    animal_df['breed_type'] = animal_df['breed_intake'].apply(lambda x: 'Mix' if 'Mix' in str(x) else 'Pure')
    animal_df['color_group'] = animal_df['color_intake'].apply(simplify_color)
    animal_df['intake_condition_group'] = animal_df['intake_condition'].apply(group_intake_condition)

    animal_df.rename(columns={'animal_type_intake': 'animal_type'}, inplace=True)
    animal_df.drop(columns=['animal_type_outcome'], inplace=True)

    return animal_df

# --- Final Cleaning ---
def final_cleaning(animal_df):
    final_drop_cols = [
        'animal_id', 'datetime_intake', 'datetime_outcome', 'datetime2', 'found_location',
        'intake_type', 'intake_condition', 'breed_intake', 'color_intake', 'monthyear_outcome',
        'date_of_birth_outcome', 'outcome_type', 'sex_upon_intake', 'sex_upon_outcome',
        'age_upon_intake', 'age_upon_outcome', 'breed_outcome', 'color_outcome',
        'reproductive_status', 'age_group_outcome'
    ]
    animal_df.drop(columns=[col for col in final_drop_cols if col in animal_df.columns], inplace=True)
    animal_df.drop(columns=['age_days_intake'], inplace=True, errors='ignore')
    return animal_df

# --- Saving ---
def save_clean_data(animal_df, filename='animal_df.csv'):
    animal_df.to_csv(filename, index=False)

# --- Helpers ---
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
    groups = {
        'Black': ['black', 'ebony', 'charcoal'],
        'White': ['white', 'cream', 'ivory', 'pearl', 'platinum'],
        'Brown': ['brown', 'brindle', 'chocolate', 'mocha', 'mahogany'],
        'Grey': ['gray', 'grey', 'blue', 'silver', 'slate', 'pewter', 'ash'],
        'Gold': ['gold', 'tan', 'yellow', 'fawn', 'buff', 'sandy'],
        'Red': ['red', 'ginger', 'orange', 'copper', 'auburn'],
        'Cream': ['cream', 'beige', 'off-white']
    }
    for group, keywords in groups.items():
        if any(word in color for word in keywords):
            return group
    if 'tabby' in color:
        return 'Tabby'
    if 'merle' in color:
        return 'Merle'
    if 'tricolor' in color or ('black' in color and 'tan' in color and 'white' in color):
        return 'Tricolor'
    if any(word in color for word in ['spotted', 'speckled', 'ticked', 'roan', 'freckled']):
        return 'Spotted'
    if '/' in color or 'and' in color:
        return 'Multicolor'
    return 'Other'

def group_intake_condition(condition):
    condition = str(condition).lower()
    if condition == 'normal':
        return 'Healthy/Normal'
    if condition in ['injured', 'neurologic', 'med attn', 'med urgent']:
        return 'Injured'
    if condition in ['sick', 'medical', 'parvo', 'panleuk', 'congenital']:
        return 'Sick/Medical'
    if condition in ['behavior', 'feral']:
        return 'Behavioral Issues'
    if condition in ['neonatal', 'nursing', 'aged', 'pregnant']:
        return 'Life Stage/Developmental'
    return 'Other/Unknown'
