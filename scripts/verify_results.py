import os
import pandas as pd
import joblib
import tensorflow as tf
from google.cloud import bigquery
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def run_backtest():
    # 1. Load Model & Scaler
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = tf.keras.models.load_model(os.path.join(project_root, "models", "best_72h_model_ann.keras"))
    scaler = joblib.load(os.path.join(project_root, "models", "scaler.joblib"))

    # 2. Fetch a specific 72-hour window from the past to compare
    client = bigquery.Client.from_service_account_json(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
    
    # We pick a row from 4 days ago to see how it predicted the 'future' back then
    query = f"""
        SELECT * FROM `{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}`
        WHERE timestamp < DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 4 DAY)
        ORDER BY timestamp DESC LIMIT 1
    """
    test_row = client.query(query).to_dataframe()

    # 3. Prepare features and get ACTUALS
    target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    actual_values = test_row[target_cols].values.flatten()
    
    feature_cols = [col for col in test_row.columns if col not in target_cols + ['timestamp', 'city', 'aqi']]
    features = test_row[feature_cols]

    # 4. Predict
    features_scaled = scaler.transform(features)
    predicted_values = model.predict(features_scaled)[0]

    # 5. Compare & Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual AQI', color='blue', linewidth=2)
    plt.plot(predicted_values, label='Predicted AQI', color='red', linestyle='--', linewidth=2)
    plt.title(f"Model Verification: Predicted vs Actual (72-Hour Window)")
    plt.xlabel("Hours into the Future")
    plt.ylabel("AQI (PM2.5)")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(project_root, "verification_plot.png")
    plt.savefig(save_path)
    print(f"✅ Verification plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_backtest()