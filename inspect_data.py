import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "thermal.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print("Columns:", df.columns.tolist())
    print("\nUnique values for WFR:")
    print(df["WFR"].unique())
    print("\nUnique values for ORIENTATION:")
    print(df["ORIENTATION"].unique())
except Exception as e:
    print(f"Error: {e}")
