# import os
# import pandas as pd
# from datetime import datetime, timedelta
# from google.cloud import bigquery
# from src.feature_engineering import compute_features
# from dotenv import load_dotenv

# load_dotenv()

# def run_feature_pipeline():
#     # 1. Setup Time Window (Fetch only the last 24-48 hours)
#     today = datetime.now().strftime('%Y-%m-%d')
#     yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    
#     print(f"🔄 Updating Feature Store for Karachi ({yesterday} to {today})...")

#     # 2. Fetch Live Data (using your existing fetch logic)
#     # Note: Use the same lat/lon (24.8607, 67.0011) as the backfill
#     df_raw = fetch_weather_and_aq(24.8607, 67.0011, yesterday, today)
    
#     # 3. Compute Features (Reuse your logic for consistency!)
#     df_new_features = compute_features(df_raw)
#     df_new_features['city'] = "Karachi"

#     # 4. Upsert into BigQuery
#     client = bigquery.Client.from_service_account_json(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
#     table_id = f"{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"

#     # We use WRITE_APPEND because we want to add NEW rows to the existing history
#     job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    
#     print(f"📤 Appending latest features to {table_id}...")
#     client.load_table_from_dataframe(df_new_features, table_id, job_config=job_config).result()
#     print("✅ Feature Store updated successfully.")

# if __name__ == "__main__":
#     run_feature_pipeline()
# import os
# import sys
# import pandas as pd
# import requests
# from datetime import datetime, timedelta
# from pymongo import MongoClient, UpdateOne
# from pymongo.errors import ConfigurationError
# from dotenv import load_dotenv

# # --- PATH FIX ---
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# load_dotenv(os.path.join(project_root, '.env'))

# from src.feature_engineering import compute_features

# def fetch_weather_and_aq(lat, lon, start, end):
#     print(f"📡 Fetching data for Karachi ({start} to {end})...")
#     # Note: Open-Meteo Archive can have a 2-5 day delay for the most recent 'Archive' data
#     aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}"
#     w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
#     aq_res = requests.get(aq_url).json()
#     w_res = requests.get(w_url).json()

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
#     # We fetch 10 days back because compute_features needs a 72h (3 day) 
#     # future window to create targets. 10 - 3 = 7 days of valid data.
#     today = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d') # Archive delay buffer
#     start_date = (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d')
    
#     print(f"🔄 Fetching 10-day chunk ({start_date} to {today})...")

#     # 3. Fetch & Compute
#     df_raw = fetch_weather_and_aq(24.8607, 67.0011, start_date, today)
    
#     # Feature engineering deletes rows that don't have 72h of 'future' data
#     df_new_features = compute_features(df_raw)
    
#     if df_new_features.empty:
#         print("⚠️ Warning: After feature engineering, 0 records remain. Try a larger date range.")
#         return

#     df_new_features['city'] = "Karachi"
#     df_new_features['timestamp'] = pd.to_datetime(df_new_features['timestamp'])

#     # 4. Bulk Upsert
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
#         print(f"✅ Mongo Update: {result.upserted_count} new docs, {result.modified_count} updated docs.")

# if __name__ == "__main__":
#     run_feature_pipeline()


import os
import sys
import pandas as pd
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

from src.feature_engineering import compute_features

def fetch_weather_and_aq(lat, lon, start, end):
    print(f"📡 Fetching data for Karachi ({start} to {end})...")
    # Air Quality API (Usually real-time)
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}"
    # Weather Archive API
    w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
    aq_res = requests.get(aq_url).json()
    w_res = requests.get(w_url).json()

    # Safety check if API fails
    if "hourly" not in aq_res or "hourly" not in w_res:
        print("❌ Error: API response invalid.")
        return pd.DataFrame()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
        "aqi": aq_res["hourly"]["pm2_5"],
        "temp": w_res["hourly"]["temperature_2m"],
        "humidity": w_res["hourly"]["relative_humidity_2m"],
        "wind_speed": w_res["hourly"]["wind_speed_10m"]
    })

def run_feature_pipeline():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        print("🔌 Connecting to MongoDB...")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        db = client[db_name]
        collection = db[col_name]
        client.admin.command('ping')
        print("✅ Connection Successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # --- UPDATED TIME WINDOW ---
    # Fetching 14 days ensures we have:
    # 1. 6 hours of history for 'aqi_rolling_6h'
    # 2. 72 hours of future data for 'target_aqi_Xh'
    today = datetime.now().strftime('%Y-%m-%d') 
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    
    print(f"🔄 Syncing 14-day window ({start_date} to {today})...")

    # 3. Fetch & Compute
    df_raw = fetch_weather_and_aq(24.8607, 67.0011, start_date, today)
    
    if df_raw.empty:
        return

    # This calls your NEW compute_features (with is_winter, rolling_avg, etc.)
    df_new_features = compute_features(df_raw)
    
    if df_new_features.empty:
        print("⚠️ Warning: After feature engineering, 0 records remain. This usually means the API data doesn't have enough future/past buffer.")
        return

    df_new_features['city'] = "Karachi"
    df_new_features['timestamp'] = pd.to_datetime(df_new_features['timestamp'])

    # 4. Bulk Upsert (Using $set to ensure new columns are added to existing docs)
    print(f"📤 Syncing {len(df_new_features)} records to MongoDB...")
    ops = [
        UpdateOne(
            filter={"timestamp": row["timestamp"], "city": row["city"]},
            update={"$set": row},
            upsert=True
        ) for row in df_new_features.to_dict('records')
    ]

    if ops:
        result = collection.bulk_write(ops)
        print(f"✅ Mongo Update: {result.upserted_count} new docs, {result.modified_count} updated/merged.")

if __name__ == "__main__":
    run_feature_pipeline()