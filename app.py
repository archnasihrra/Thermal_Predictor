# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Config
st.set_page_config(
    page_title="Thermal Comfort Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3D59;
    }
    .stMetricLabel {
        color: #172B4D !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load artifacts
# Use st.cache_resource to avoid reloading on every interaction
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_DIR / "model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        encoders = joblib.load(MODEL_DIR / "encoders.pkl")
        return model, scaler, encoders
    except Exception as e:
        return None, None, None

model, scaler, encoders = load_artifacts()

if not model:
    st.error("❌ Critical Error: Model files could not be loaded. Please ensure `train.py` has been executed successfully.")
    st.stop()

orientation_map = encoders["orientation_map"]
wfr_map = encoders["wfr_map"]
ashrae_label_map = {
    -3: "Cold 🥶",
    -2: "Cool ❄️",
    -1: "Slightly Cool 🌬️",
     0: "Neutral 😌",
     1: "Slightly Warm 🌤️",
     2: "Warm ☀️",
     3: "Hot 🔥"
}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Design Parameters")
    st.markdown("Adjust the building parameters below to simulate thermal comfort.")
    
    st.divider()
    
    orientation = st.selectbox(
        "🧭 Orientation via Compass", 
        sorted(orientation_map.keys()),
        help="Direction the facade is facing."
    )
    
    wfr = st.selectbox(
        "🪟 Window-to-Floor Ratio", 
        list(wfr_map.keys()),
        help="Percentage of floor area dedicated to glazing."
    )
    
    st.divider()
    
    temp = st.slider(
        "🌡️ Indoor Temp (°F)", 
        60, 95, 75,
        help="Current indoor dry-bulb temperature."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        space = st.number_input(
            "👥 Sq.Ft/Person", 
            50, 1000, 150
        )
    with col2:
        power = st.number_input(
            "⚡ Power (kWh)", 
            0.0, 2000.0, 500.0
        )

# --- Main Content ---

st.title("🌡️ Thermal Comfort Predictor")
st.markdown("### Decision Support System for Building Performance")
st.markdown("---")

# Main Action Button
if st.button("🔮 Analyze Comfort Level", type="primary", use_container_width=True):
    
    # 1. Prepare Data
    or_val = orientation_map[orientation]
    wfr_val = wfr_map[wfr]
    
    X = np.array([[
        or_val,
        wfr_val,
        temp,
        space,
        power
    ]])

    # 2. Predict
    X_scaled = scaler.transform(X)
    prediction_idx = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    # 3. Interpret
    result_label = ashrae_label_map.get(prediction_idx, "Unknown")
    
    # 4. Display Results
    st.markdown("#### Analysis Results")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        # color-coding the result card
        box_color = "#f0f2f6"
        if prediction_idx > 0: box_color = "#ffebee" # warm/red
        elif prediction_idx < 0: box_color = "#e3f2fd" # cool/blue
        else: box_color = "#e8f5e9" # neutral/green
        
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
            <h4 style="margin:0; color: #555;">Predicted Sensation</h4>
            <h2 style="margin:10px 0; color: #000;">{result_label}</h2>
            <p style="margin:0; color: #777;">ASHRAE Score: <strong>{prediction_idx}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_res2:
        st.caption("Confidence Distribution")
        # Create a clean dataframe for the bar chart
        # We need to map the model's classes_ (which are -3 to 3) to their probabilities
        classes = model.classes_
        probs = prediction_proba
        
        prob_df = pd.DataFrame({
            "Sensation": [ashrae_label_map[c] for c in classes],
            "Probability": probs
        })
        
        st.bar_chart(
            prob_df.set_index("Sensation"),
            color="#1E3D59",
            height=200
        )

else:
    st.info("👈 Configure the parameters in the sidebar and click **Analyze** to see the prediction.")

    st.markdown("#### About this Tool")
    st.markdown("""
    This application uses a **Gradient Boosting Machine (GBM)** model to predict occupant thermal comfort based on:
    *   **Architectural factors**: Orientation & Window ratios.
    *   **Environmental factors**: Indoor temperature.
    *   **Usage factors**: Occupancy density & Power consumption.
    """)
