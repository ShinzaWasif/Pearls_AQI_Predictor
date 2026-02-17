import os
import sys
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# --- PATH & ENV CONFIG ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

# DagsHub/MLflow Credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(dagshub_url)

def run_inference():
    # 1. LOAD CALIBRATED MODEL AND SCALER FROM REGISTRY
    try:
        # Note: Name matches the new training script
        model_name = "AQI_72h_Karachi_Calibrated" 
        model_uri = f"models:/{model_name}/latest"
        
        print(f"Connecting to DagsHub Registry: {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # --- FETCH SCALER FROM MLFLOW ARTIFACTS ---
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        run_id = latest_version.run_id
        
        print(f"Downloading scaler from Run ID: {run_id}...")
        scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.joblib")
        scaler = joblib.load(scaler_path)
        
        print("Calibrated Model and Remote Scaler loaded successfully!")
    except Exception as e:
        print(f"ERROR connecting to Registry/Artifacts: {e}")
        return

    # 2. Get the PRE-COMPUTED row from MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    # We fetch the absolute latest row (which was calibrated by Feature Pipeline)
    latest_row = collection.find_one({"city": "Karachi"}, sort=[("timestamp", -1)])
    
    if not latest_row:
        print("No data found in MongoDB.")
        return

    # 3. Prepare Features
    # Ensure we use exactly what the scaler was trained on
    expected_features = list(scaler.feature_names_in_)
    input_df = pd.DataFrame([latest_row])
    
    # Safety Check: Fill missing features with 0 if they don't exist in MongoDB
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Filter and reorder columns to match training order
    input_df = input_df[expected_features]

    # 4. PREDICT
    print(f"Generating forecast for {latest_row['timestamp']}...")
    input_scaled = scaler.transform(input_df)
    hourly_preds = model.predict(input_scaled)
    
    # Flatten if model returns a 2D array (Batch size 1, 72 hours)
    if len(hourly_preds.shape) > 1:
        hourly_preds = hourly_preds[0]

    # Post-process: AQI cannot be negative
    hourly_preds = np.maximum(hourly_preds, 0)

    # 5. DISPLAY & SAVE
    print("\n" + "="*50)
    print(f"KARACHI CALIBRATED FORECAST (1.42x)")
    print(f"Data Timestamp: {latest_row['timestamp']}")
    print("="*50)
    
    daily_avgs = [
        ("Next 24h", np.mean(hourly_preds[0:24])),
        ("24h - 48h", np.mean(hourly_preds[24:48])),
        ("48h - 72h", np.mean(hourly_preds[48:72]))
    ]

    for label, avg in daily_avgs:
        # Updated classification for Karachi context
        status = "Good" if avg <= 50 else "Moderate" if avg <= 100 else "Unhealthy"
        if avg > 150: status = "Hazardous"
        print(f"{label:12} | Avg AQI: {avg:6.2f} | [{status}]")

    # Update MongoDB with the new predictions
    update_payload = {
        "is_predicted": True,
        "predicted_72h": hourly_preds.tolist(),
        "prediction_run_at": datetime.now(),
        "registry_model_version": model_uri,
        "calibration_factor": 1.42
    }

    collection.update_one({"_id": latest_row["_id"]}, {"$set": update_payload})
    print(f"\nPrediction cycle complete. MongoDB updated.")

if __name__ == "__main__":
    run_inference()