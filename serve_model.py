from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ✅ Load the model from correct path
model = joblib.load("src/saved_models/stacking_Logistic_Regression_combination_1.pkl")

# ✅ Define the input schema (must match your model’s expected features)
class InputData(BaseModel):
    age_days_outcome: float
    los_at_shelter: float
    animal_type: int
    color_group: int
    breed_type: int
    intake_condition_group: int
    month_of_outcome: int
    age_group_intake: int
    has_name: int
    is_fixed: int

# ✅ Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 Stacking model prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data.dict()])

    # ✅ Reorder columns to match training order
    expected_order = [
        'animal_type',
        'has_name',
        'age_days_outcome',
        'age_group_intake',
        'los_at_shelter',
        'month_of_outcome',
        'is_fixed',
        'breed_type',
        'color_group',
        'intake_condition_group'
    ]
    df = df[expected_order]

    # Run prediction
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

