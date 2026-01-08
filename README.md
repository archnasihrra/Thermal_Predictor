# Thermal Comfort Predictor

This application predicts building occupant thermal comfort based on architectural and environmental parameters using a machine learning model (Gradient Boosting).

## 🌐 Live Demo
Check out the deployed application: **[Thermal Comfort Predictor](https://thethermalpredictor.streamlit.app/)**

## 📂 Project Structure

- **`app.py`**: The main application file (Streamlit). Run this to launch the interface.
- **`train.py`**: The training script. Run this to retrain the model if you have new data in `data/thermal.csv`.
- **`models/`**: Contains the trained model artifacts (`model.pkl`, `scaler.pkl`, `encoders.pkl`).
- **`data/`**: Directory for the input dataset.
- **`requirements.txt`**: List of Python dependencies.

## 🚀 Setup & Installation

1.  **Install Python** (Version 3.10 or higher recommended).
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Running the App

To start the interface:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🧠 Retraining the Model (Optional)

If the underlying data changes, you can retrain the model:
1.  Place the updated `thermal.csv` in the `data/` folder.
2.  Run the training script:
    ```bash
    python train.py
    ```
3.  Restart the app to load the new model.
