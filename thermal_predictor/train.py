# train.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "thermal.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Rename columns to match expected schema
column_mapping = {
    "ORIENTATION": "orientation",
    "WFR": "wfr",
    "I/D TEMP. in F": "indoor_temp_f",
    "NO of Occupant sf/person ": "sf_per_person",
    "POWER Consum. In kwh": "power_kwh",
    "ASHRAE 7-Point ": "ashrae_7_point"
}
df = df.rename(columns=column_mapping)

# Drop rows with missing target or features if critical (though GBDT handles some NaNs, Sklearn GBDT might not)
df = df.dropna(subset=["ashrae_7_point"])

# Encoding Orientation
# We map all observed values to integers.
# Observed: ['NORTH' 'NNW' 'NW' 'WNW' 'WEST' 'WSW' 'SW' 'SSW' 'SOUTH' 'SSE' 'SE' 'ESE' 'EAST' 'ENE' 'NE' 'NNE']
# We'll create a sorted list to verify consistency.
orientation_labels = sorted(df["orientation"].dropna().unique())
orientation_map = {label: idx for idx, label in enumerate(orientation_labels)}
df["orientation"] = df["orientation"].map(orientation_map)

# Encoding WFR
# Observed: ['>30%' '<10%' '16.80%']
# Logic: <20% -> 0, 20-30% -> 1, >30% -> 2
def map_wfr(val):
    val_str = str(val).strip()
    if val_str == ">30%":
        return 2
    elif val_str == "<10%" or val_str == "16.80%":
        return 0 # <20%
    elif val_str == "<20%":
         return 0
    elif val_str == "20-30%":
        return 1
    return 0 # Default fallback

df["wfr"] = df["wfr"].apply(map_wfr)

X = df[
    ["orientation", "wfr", "indoor_temp_f",
     "sf_per_person", "power_kwh"]
]

y = df["ashrae_7_point"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Training model on {len(df)} samples...")
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_scaled, y)

# Save artifacts
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# Save encoders to ensure consistency in app
encoders = {
    "orientation_map": orientation_map,
    "wfr_map": {
        "<20% (< 10% or 16.8%)": 0,
        "20-30%": 1,
        ">30%": 2
    }
}
joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

print("✅ Training completed successfully")
