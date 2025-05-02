from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import joblib
import os

# --- Load saved models and encoders ---
model = joblib.load("src/saved_models/stacking_Logistic_Regression_combination_1.pkl")
input_label_encoders = joblib.load("src/saved_models/input_label_encoders.pkl")
label_encoder_y = joblib.load("src/saved_models/label_encoder_y.pkl")

# --- Dynamically create dropdown enums ---
def create_enum(name, labels):
    return Enum(name, {label.replace(" ", "_").replace("/", "_").lower(): int(code)
                       for code, label in enumerate(labels)})

AnimalType = create_enum("AnimalType", input_label_encoders["animal_type"].classes_)
AgeGroupIntake = create_enum("AgeGroupIntake", input_label_encoders["age_group_intake"].classes_)
MonthOfOutcome = create_enum("MonthOfOutcome", input_label_encoders["month_of_outcome"].classes_)
BreedType = create_enum("BreedType", input_label_encoders["breed_type"].classes_)
ColorGroup = create_enum("ColorGroup", input_label_encoders["color_group"].classes_)
IntakeConditionGroup = create_enum("IntakeConditionGroup", input_label_encoders["intake_condition_group"].classes_)

class HasName(int, Enum):
    no = 0
    yes = 1

class IsFixed(int, Enum):
    no = 0
    yes = 1

# --- Input schema ---
class InputData(BaseModel):
    age_days_outcome: float
    los_at_shelter: float
    animal_type: AnimalType
    has_name: HasName
    age_group_intake: AgeGroupIntake
    month_of_outcome: MonthOfOutcome
    is_fixed: IsFixed
    breed_type: BreedType
    color_group: ColorGroup
    intake_condition_group: IntakeConditionGroup

# --- API setup ---
app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 Stacking model prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert enums to values
    features = {
        "animal_type": data.animal_type.value,
        "has_name": data.has_name.value,
        "age_days_outcome": data.age_days_outcome,
        "age_group_intake": data.age_group_intake.value,
        "los_at_shelter": data.los_at_shelter,
        "month_of_outcome": data.month_of_outcome.value,
        "is_fixed": data.is_fixed.value,
        "breed_type": data.breed_type.value,
        "color_group": data.color_group.value,
        "intake_condition_group": data.intake_condition_group.value
    }

    df = pd.DataFrame([features])

    prediction = model.predict(df)
    readable = label_encoder_y.inverse_transform([prediction[0]])[0]

    return {
        "prediction_class": int(prediction[0]),
        "outcome_label": readable
    }
@app.get("/info")
def get_label_mappings():
    """Return human-readable label mappings for all encoded fields."""
    mappings = {}

    for field, encoder in input_label_encoders.items():
        mappings[field] = {
            int(code): label for code, label in enumerate(encoder.classes_)
        }

    mappings["has_name"] = {0: "No", 1: "Yes"}
    mappings["is_fixed"] = {0: "No", 1: "Yes"}

    return mappings
