import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

# This imports the updated logic with EPA formula and Smog Index
from src.feature_engineering import compute_features

# --- ROBUST SESSION CONFIG ---
def get_robust_session():
    """Creates a requests session with built-in retries and backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_weather_and_aq(lat, lon, start, end):
    """Fetches historical PM2.5 and Weather data with calibration."""
    session = get_robust_session()
    
    aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?"
              f"latitude={lat}&longitude={lon}&hourly=pm2_5&"
              f"start_date={start}&end_date={end}")
    
    w_url = (f"https://archive-api.open-meteo.com/v1/archive?"
             f"latitude={lat}&longitude={lon}&hourly=temperature_2m,"
             f"relative_humidity_2m,wind_speed_10m&"
             f"start_date={start}&end_date={end}")
    
    try:
        aq_res = session.get(aq_url, timeout=60).json()
        w_res = session.get(w_url, timeout=60).json()

        if "hourly" not in aq_res or "hourly" not in w_res:
            print(f"Warning: Missing hourly data for range {start} to {end}")
            return pd.DataFrame()

        # --- APPLY 1.42 CALIBRATION IMMEDIATELY ---
        pm25_calibrated = [val * 1.42 if val is not None else None for val in aq_res["hourly"]["pm2_5"]]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
            "aqi": pm25_calibrated, 
            "temp": w_res["hourly"]["temperature_2m"],
            "humidity": w_res["hourly"]["relative_humidity_2m"],
            "wind_speed": w_res["hourly"]["wind_speed_10m"]
        })
        return df
    except Exception as e:
        print(f"API Request failed for {start} to {end}: {e}")
        return pd.DataFrame()

def run_backfill():
    # 1. Configuration for Batching
    lat, lon = 24.8607, 67.0011
    
    # --- DYNAMIC DATE LOGIC ---
    today = datetime.now()
    
    # Calculate 6 months ago
    six_months_ago = today - timedelta(days=180) 
    # Adjust to the 1st of that month
    start_date = six_months_ago.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # End date is 8 days ago
    end_date = today - timedelta(days=8)
    
    all_chunks = []
    current_start = start_date

    print(f"Starting Dynamic Backfill (1.42x): {start_date.date()} to {end_date.date()}")

    # 2. Fetch Data in 30-Day Chunks
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=30), end_date)
        
        s_str = current_start.strftime("%Y-%m-%d")
        e_str = current_end.strftime("%Y-%m-%d")
        
        print(f"Fetching chunk: {s_str} to {e_str}...")
        df_chunk = fetch_weather_and_aq(lat, lon, s_str, e_str)
        
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
            print(f"Chunk loaded ({len(df_chunk)} rows)")
        
        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        print("Warning: No data fetched. Check API or internet connection.")
        return

    # Combine all chunks
    df_raw = pd.concat(all_chunks).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    # 3. Compute Features
    print(f"Computing features for {len(df_raw)} total records...")
    df_final = compute_features(df_raw, training_mode=True)
    
    if df_final.empty:
        print("Warning: No records after processing.")
        return

    df_final['city'] = "Karachi"
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])

    # 4. MongoDB Sync
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        print(f"Connecting to MongoDB: {db_name}...")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[col_name]
        
        print(f"Syncing {len(df_final)} calibrated records...")
        
        ops = [
            UpdateOne(
                filter={"timestamp": row["timestamp"], "city": row["city"]},
                update={"$set": row.to_dict()},
                upsert=True
            ) for _, row in df_final.iterrows()
        ]

        if ops:
            result = collection.bulk_write(ops)
            print(f"Calibrated Backfill Complete!")
            print(f"Stats: {result.upserted_count} new records, {result.modified_count} updated.")
            
    except Exception as e:
        print(f"MongoDB Error: {e}")

if __name__ == "__main__":
    run_backfill()