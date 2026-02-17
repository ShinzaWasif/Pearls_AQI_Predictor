import os
import sys
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv()

# Ensure this path matches where your updated compute_features is stored
from src.feature_engineering import compute_features

def fetch_weather_and_aq(lat, lon, start, end):
    """
    Fetches raw data from Open-Meteo APIs.
    """
    print(f"Syncing Karachi data (Zero-Lag Rolling Window)...")
    
    # Air Quality API: PM2.5 data
    aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?"
              f"latitude={lat}&longitude={lon}&hourly=pm2_5&"
              f"past_days=7&forecast_days=4&timezone=auto")
    
    # Weather Forecast API
    weather_url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={lat}&longitude={lon}&hourly=temperature_2m,"
                   f"relative_humidity_2m,wind_speed_10m&"
                   f"past_days=7&forecast_days=3&timezone=auto")

    try:
        aq_res = requests.get(aq_url).json()
        w_res = requests.get(weather_url).json()

        if "hourly" not in aq_res or "hourly" not in w_res:
            error_msg = aq_res.get('reason', w_res.get('reason', 'Unknown error'))
            print(f"API Error: {error_msg}")
            return pd.DataFrame()

        df_aq = pd.DataFrame({
            "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
            "aqi": aq_res["hourly"]["pm2_5"]
        })
        
        df_w = pd.DataFrame({
            "timestamp": pd.to_datetime(w_res["hourly"]["time"]),
            "temp": w_res["hourly"]["temperature_2m"],
            "humidity": w_res["hourly"]["relative_humidity_2m"],
            "wind_speed": w_res["hourly"]["wind_speed_10m"]
        })

        df = pd.merge(df_aq, df_w, on="timestamp", how="inner")
        df = df.dropna(subset=['aqi', 'temp', 'humidity', 'wind_speed'])

        # --- 1.42 CALIBRATION STEP ---
        print("Applying 1.42 Calibration Factor to raw PM2.5...")
        df['aqi'] = df['aqi'] * 1.42
        
        return df.sort_values("timestamp")

    except Exception as e:
        print(f"Fetch failed: {e}")
        return pd.DataFrame()

def run_feature_pipeline():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        db = client[db_name]
        collection = db[col_name]
        print("MongoDB Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # --- 14-DAY SYNC WINDOW ---
    today = datetime.now().strftime('%Y-%m-%d') 
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    # --- FETCH DATA ---
    df_raw = fetch_weather_and_aq(24.8607, 67.0011, start_date, today)
    
    if df_raw.empty:
        return

    # --- TRANSFORM DATA ---
    df_transformed = compute_features(df_raw, training_mode=False)
    
    if df_transformed.empty:
        print("Warning: Post-processing resulted in 0 records.")
        return

    print(f"Processing complete: {df_transformed['timestamp'].min()} to {df_transformed['timestamp'].max()}")

    df_transformed['city'] = "Karachi"
    df_transformed['timestamp'] = pd.to_datetime(df_transformed['timestamp'])

    # BULK UPSERT
    print(f"Syncing {len(df_transformed)} calibrated records to MongoDB...")
    ops = [
        UpdateOne(
            filter={"timestamp": row["timestamp"], "city": row["city"]},
            update={"$set": row.to_dict()},
            upsert=True
        ) for _, row in df_transformed.iterrows()
    ]

    if ops:
        result = collection.bulk_write(ops)
        print(f"Sync Successful: {result.upserted_count} new, {result.modified_count} updated.")

if __name__ == "__main__":
    run_feature_pipeline()