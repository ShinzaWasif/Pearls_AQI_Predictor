# import os
# import requests
# import pandas as pd
# from google.cloud import bigquery
# from src.feature_engineering import compute_features
# from dotenv import load_dotenv

# load_dotenv()

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
#     # 1. Fetch Raw Data (Using a wide range for training)
#     # Note: Ensure start/end dates are valid for your project
#     df = fetch_weather_and_aq(24.8607, 67.0011, "2024-01-01", "2025-12-25")
    
#     # 2. Compute Features (This now creates the 72h targets and weather leads)
#     print("🛠️ Computing multi-output features and 72h leads...")
#     df_features = compute_features(df)
#     df_features['city'] = "Karachi"

#     # --- CRITICAL CHANGE START ---
#     # Instead of hardcoding ordered_columns, we keep EVERY column created 
#     # by compute_features. This includes target_aqi_1h...72h, temp_f_1h... etc.
#     df_final = df_features.copy()
#     # --- CRITICAL CHANGE END ---

#     # 3. Store in BigQuery
#     client = bigquery.Client.from_service_account_json(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
    
#     project_id = os.getenv('GCP_PROJECT_ID')
#     dataset_id = os.getenv('BQ_DATASET_ID')
#     table_id = f"{project_id}.{dataset_id}.{os.getenv('BQ_TABLE_ID')}"

#     # Safety check: Create dataset if it doesn't exist
#     dataset_ref = client.dataset(dataset_id)
#     try:
#         client.get_dataset(dataset_ref)
#     except Exception:
#         print(f"Creating dataset {dataset_id}...")
#         client.create_dataset(bigquery.Dataset(dataset_ref))

#     # WRITE_TRUNCATE is important here because it resets the table schema 
#     # to accommodate the 200+ new columns.
#     job_config = bigquery.LoadJobConfig(
#         write_disposition="WRITE_TRUNCATE", 
#         autodetect=True
#     )
    
#     print(f"📤 Uploading {len(df_final.columns)} columns to BigQuery Warehouse...")
#     client.load_table_from_dataframe(df_final, table_id, job_config=job_config).result()
#     print(f"✅ Successfully stored 3-day forecasting dataset in BigQuery!")

# if __name__ == "__main__":
#     run_pipeline()


import os
import sys
import requests
import pandas as pd
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# --- PATH FIX ---
# Ensures the script can find 'src' regardless of where it's run from
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

from src.feature_engineering import compute_features

def fetch_weather_and_aq(lat, lon, start, end):
    print(f"📡 Fetching historical data for Karachi ({start} to {end})...")
    # Air Quality API
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&start_date={start}&end_date={end}"
    # Weather Archive API
    w_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&start_date={start}&end_date={end}"
    
    aq_res = requests.get(aq_url).json()
    w_res = requests.get(w_url).json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(aq_res["hourly"]["time"]),
        "aqi": aq_res["hourly"]["pm2_5"],
        "temp": w_res["hourly"]["temperature_2m"],
        "humidity": w_res["hourly"]["relative_humidity_2m"],
        "wind_speed": w_res["hourly"]["wind_speed_10m"]
    })
    return df

def run_pipeline():
    # 1. Fetch Raw Data 
    # To get ~5 months of data, we go from August 2025 to late Jan 2026
    df = fetch_weather_and_aq(24.8607, 67.0011, "2025-08-01", "2026-01-24")
    
    # 2. Compute Features
    print("🛠️ Computing multi-output features and 72h leads...")
    df_features = compute_features(df)
    df_features['city'] = "Karachi"
    
    # Making sure timestamp is clean for Mongo
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

    # Keeping every column as requested
    df_final = df_features.copy()

    # 3. Store in MongoDB (Replacing BigQuery)
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        print(f"🔌 Connecting to MongoDB Cluster...")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[col_name]
        
        # 4. Prepare Bulk Upsert
        print(f"📤 Preparing to upload {len(df_final)} records with {len(df_final.columns)} columns...")
        
        ops = [
            UpdateOne(
                filter={"timestamp": row["timestamp"], "city": row["city"]},
                update={"$set": row},
                upsert=True
            ) for row in df_final.to_dict('records')
        ]

        if ops:
            # We use bulk_write because with 3,000+ records, individual inserts are too slow
            result = collection.bulk_write(ops)
            print(f"✅ Successfully stored 5-month dataset in MongoDB!")
            print(f"📊 Stats: {result.upserted_count} new records, {result.modified_count} updated.")
            
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")

if __name__ == "__main__":
    run_pipeline()