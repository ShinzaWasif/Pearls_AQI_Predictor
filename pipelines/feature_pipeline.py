# import os
# import sys
# import pandas as pd
# import requests
# from datetime import datetime, timedelta
# from pymongo import MongoClient, UpdateOne
# from dotenv import load_dotenv

# # --- PATH FIX ---
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# # load_dotenv(os.path.join(project_root, '.env'))
# load_dotenv()


# from src.feature_engineering import compute_features

# def fetch_weather_and_aq(lat, lon, start, end):
#     print(f"📡 Fetching data for Karachi ({start} to {end})...")
#     # Air Quality API (Usually real-time)
#     aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}"
#     # Weather Archive API
#     w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
#     aq_res = requests.get(aq_url).json()
#     w_res = requests.get(w_url).json()

#     # Safety check if API fails
#     if "hourly" not in aq_res or "hourly" not in w_res:
#         print("❌ Error: API response invalid.")
#         return pd.DataFrame()

#     return pd.DataFrame({
#         "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
#         "aqi": aq_res["hourly"]["pm2_5"],
#         "temp": w_res["hourly"]["temperature_2m"],
#         "humidity": w_res["hourly"]["relative_humidity_2m"],
#         "wind_speed": w_res["hourly"]["wind_speed_10m"]
#     })

# def run_feature_pipeline():
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     col_name = os.getenv("MONGO_COLLECTION_NAME")

#     try:
#         print("🔌 Connecting to MongoDB...")
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
#         db = client[db_name]
#         collection = db[col_name]
#         client.admin.command('ping')
#         print("✅ Connection Successful!")
#     except Exception as e:
#         print(f"❌ Connection failed: {e}")
#         return

#     # --- UPDATED TIME WINDOW ---
#     # Fetching 14 days ensures we have:
#     # 1. 6 hours of history for 'aqi_rolling_6h'
#     # 2. 72 hours of future data for 'target_aqi_Xh'
#     today = datetime.now().strftime('%Y-%m-%d') 
#     start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
#     print(f"🔄 Syncing 14-day window ({start_date} to {today})...")

#     # 3. Fetch & Compute
#     df_raw = fetch_weather_and_aq(24.8607, 67.0011, start_date, today)
    
#     if df_raw.empty:
#         return

#     # This calls your NEW compute_features (with is_winter, rolling_avg, etc.)
#     df_new_features = compute_features(df_raw)
    
#     if df_new_features.empty:
#         print("⚠️ Warning: After feature engineering, 0 records remain. This usually means the API data doesn't have enough future/past buffer.")
#         return

#     df_new_features['city'] = "Karachi"
#     df_new_features['timestamp'] = pd.to_datetime(df_new_features['timestamp'])

#     # 4. Bulk Upsert (Using $set to ensure new columns are added to existing docs)
#     print(f"📤 Syncing {len(df_new_features)} records to MongoDB...")
#     ops = [
#         UpdateOne(
#             filter={"timestamp": row["timestamp"], "city": row["city"]},
#             update={"$set": row},
#             upsert=True
#         ) for row in df_new_features.to_dict('records')
#     ]

#     if ops:
#         result = collection.bulk_write(ops)
#         print(f"✅ Mongo Update: {result.upserted_count} new docs, {result.modified_count} updated/merged.")

# if __name__ == "__main__":
#     run_feature_pipeline()

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

# This imports the compute_features function that uses the OFFICIAL EPA formula
from src.feature_engineering import compute_features

def fetch_weather_and_aq(lat, lon, start, end):
    """
    Fetches raw PM2.5 and meteorological data. 
    PM2.5 is the input for the official USA AQI formula.
    """
    print(f"📡 Fetching data for Karachi ({start} to {end})...")
    
    # Air Quality API: Fetching PM2.5
    aq_url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?"
              f"latitude={lat}&longitude={lon}&hourly=pm2_5&"
              f"start_date={start}&end_date={end}")
    
    # Weather Archive API: Fetching Temp, Humidity, and Wind
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

        return pd.DataFrame({
            "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
            "aqi": aq_res["hourly"]["pm2_5"], # Raw PM2.5 Concentration
            "temp": w_res["hourly"]["temperature_2m"],
            "humidity": w_res["hourly"]["relative_humidity_2m"],
            "wind_speed": w_res["hourly"]["wind_speed_10m"]
        })
    except Exception as e:
        print(f"❌ API Request failed: {e}")
        return pd.DataFrame()

def run_feature_pipeline():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        db = client[db_name]
        collection = db[col_name]
        print("✅ MongoDB Connected.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # --- 14-DAY SYNC WINDOW ---
    # This buffer ensures rolling averages and future leads are fully populated
    today = datetime.now().strftime('%Y-%m-%d') 
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    df_raw = fetch_weather_and_aq(24.8607, 67.0011, start_date, today)
    
    if df_raw.empty:
        return

    # COMPUTE: Using the Official EPA Non-Linear Logic
    df_transformed = compute_features(df_raw)
    
    if df_transformed.empty:
        print("⚠️ Warning: Post-processing resulted in 0 records.")
        return

    df_transformed['city'] = "Karachi"
    df_transformed['timestamp'] = pd.to_datetime(df_transformed['timestamp'])

    # BULK UPSERT: Use $set to merge with existing data (prevents overwriting predictions)
    print(f"📤 Syncing {len(df_transformed)} records to MongoDB...")
    ops = [
        UpdateOne(
            filter={"timestamp": row["timestamp"], "city": row["city"]},
            update={"$set": row.to_dict()},
            upsert=True
        ) for _, row in df_transformed.iterrows()
    ]

    if ops:
        result = collection.bulk_write(ops)
        print(f"✅ Sync Successful: {result.upserted_count} new, {result.modified_count} updated.")

if __name__ == "__main__":
    run_feature_pipeline()