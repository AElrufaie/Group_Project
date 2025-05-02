import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocessing import (
    load_or_fetch_data,
    prepare_dataframes,
    feature_engineering,
    final_cleaning
)
from src.modeling.encoding import label_encode_columns

# Set data directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

print("📥 Loading and preparing data...")

# Base URLs (used for loading if local file missing)
base_url_1 = "https://data.austintexas.gov/resource/9t4d-g238.csv"
base_url_2 = "https://data.austintexas.gov/resource/wter-evkm.csv"

df_intake, df_outcome = load_or_fetch_data(DATA_FOLDER, base_url_1, base_url_2)
animal_df = prepare_dataframes(df_intake, df_outcome)
animal_df = feature_engineering(animal_df)
df = final_cleaning(animal_df)

print("🔤 Encoding input features...")
label_encode_cols = [
    'age_group_intake', 'breed_type', 'color_group',
    'intake_condition_group', 'animal_type', 'month_of_outcome'
]

df_encoded, label_encoders = label_encode_columns(df, label_encode_cols)

print("🎯 Encoding target labels...")
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(df_encoded["outcome_group"])

print("💾 Saving encoders to src/saved_models/ ...")

saved_dir = os.path.join(PROJECT_ROOT, "src", "saved_models")
os.makedirs(saved_dir, exist_ok=True)

joblib.dump(label_encoders, os.path.join(saved_dir, "input_label_encoders.pkl"))
joblib.dump(label_encoder_y, os.path.join(saved_dir, "label_encoder_y.pkl"))

print("✅ Encoders saved successfully.")

