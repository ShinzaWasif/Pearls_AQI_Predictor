
# import os
# import sys
# import requests
# import pandas as pd
# from pymongo import MongoClient, UpdateOne
# from dotenv import load_dotenv

# # --- PATH FIX ---
# # Ensures the script can find 'src' regardless of where it's run from
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# load_dotenv(os.path.join(project_root, '.env'))

# from src.feature_engineering import compute_features

# def fetch_weather_and_aq(lat, lon, start, end):
#     print(f"📡 Fetching historical data for Karachi ({start} to {end})...")
#     # Air Quality API
#     aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}"
#     # Weather Archive API
#     w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
#     aq_res = requests.get(aq_url).json()
#     w_res = requests.get(w_url).json()

#     df = pd.DataFrame({
#         "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
#         "aqi": aq_res["hourly"]["pm2_5"],
#         "temp": w_res["hourly"]["temperature_2m"],
#         "humidity": w_res["hourly"]["relative_humidity_2m"],
#         "wind_speed": w_res["hourly"]["wind_speed_10m"]
#     })
#     return df

# def run_pipeline():
#     # 1. Fetch Raw Data 
#     # To get ~5 months of data, we go from August 2025 to late Jan 2026
#     df = fetch_weather_and_aq(24.8607, 67.0011, "2025-08-01", "2026-01-24")
    
#     # 2. Compute Features
#     print("🛠️ Computing multi-output features and 72h leads...")
#     df_features = compute_features(df)
#     df_features['city'] = "Karachi"
    
#     # Making sure timestamp is clean for Mongo
#     df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

#     # Keeping every column as requested
#     df_final = df_features.copy()

#     # 3. Store in MongoDB (Replacing BigQuery)
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     col_name = os.getenv("MONGO_COLLECTION_NAME")

#     try:
#         print(f"🔌 Connecting to MongoDB Cluster...")
#         client = MongoClient(mongo_uri)
#         db = client[db_name]
#         collection = db[col_name]
        
#         # 4. Prepare Bulk Upsert
#         print(f"📤 Preparing to upload {len(df_final)} records with {len(df_final.columns)} columns...")
        
#         ops = [
#             UpdateOne(
#                 filter={"timestamp": row["timestamp"], "city": row["city"]},
#                 update={"$set": row},
#                 upsert=True
#             ) for row in df_final.to_dict('records')
#         ]

#         if ops:
#             # We use bulk_write because with 3,000+ records, individual inserts are too slow
#             result = collection.bulk_write(ops)
#             print(f"✅ Successfully stored 5-month dataset in MongoDB!")
#             print(f"📊 Stats: {result.upserted_count} new records, {result.modified_count} updated.")
            
#     except Exception as e:
#         print(f"❌ MongoDB Error: {e}")

# if __name__ == "__main__":
#     run_pipeline()

import os
import sys
import requests
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

# This now imports the updated logic with EPA formula and Smog Index
from src.feature_engineering import compute_features

def fetch_weather_and_aq(lat, lon, start, end):
    """Fetches historical PM2.5 and Weather data for backfilling."""
    print(f"📡 Fetching historical data for Karachi ({start} to {end})...")
    
    # 1. Air Quality API (PM2.5)
    aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?"
              f"latitude={lat}&longitude={lon}&hourly=pm2_5&"
              f"start_date={start}&end_date={end}")
    
    # 2. Weather Archive API (Met Data)
    w_url = (f"https://archive-api.open-meteo.com/v1/archive?"
             f"latitude={lat}&longitude={lon}&hourly=temperature_2m,"
             f"relative_humidity_2m,wind_speed_10m&"
             f"start_date={start}&end_date={end}")
    
    try:
        aq_res = requests.get(aq_url).json()
        w_res = requests.get(w_url).json()

        if "hourly" not in aq_res or "hourly" not in w_res:
            print("❌ Error: API response invalid.")
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
            "aqi": aq_res["hourly"]["pm2_5"], # Input for compute_features
            "temp": w_res["hourly"]["temperature_2m"],
            "humidity": w_res["hourly"]["relative_humidity_2m"],
            "wind_speed": w_res["hourly"]["wind_speed_10m"]
        })
        return df
    except Exception as e:
        print(f"❌ API Request failed: {e}")
        return pd.DataFrame()

def run_backfill():
    # 1. Fetch Raw Data (August 2025 to Present)
    # We fetch a slightly larger window to ensure the rolling/shift logic has buffers
    df_raw = fetch_weather_and_aq(24.8607, 67.0011, "2025-08-01", "2026-02-01")
    
    if df_raw.empty:
        print("⚠️ No data fetched. Check API or dates.")
        return

    # 2. Compute Features (EPA Formula, Smog Index, 72h Leads)
    print("🛠️ Computing EPA-calibrated features and 72h future targets...")
    df_final = compute_features(df_raw)
    
    df_final['city'] = "Karachi"
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])

    # 3. MongoDB Connection
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        print(f"🔌 Connecting to MongoDB: {db_name}...")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[col_name]
        
        # 4. Prepare Bulk Upsert (Using $set to avoid overwriting existing metadata)
        print(f"📤 Syncing {len(df_final)} historical records to MongoDB...")
        
        ops = [
            UpdateOne(
                filter={"timestamp": row["timestamp"], "city": row["city"]},
                update={"$set": row.to_dict()},
                upsert=True
            ) for _, row in df_final.iterrows()
        ]

        if ops:
            result = collection.bulk_write(ops)
            print(f"✅ Backfill Complete!")
            print(f"📊 Stats: {result.upserted_count} new records, {result.modified_count} updated.")
            
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")

if __name__ == "__main__":
    run_backfill()