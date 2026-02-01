# import os
# import requests
# import pandas as pd
# import joblib
# import numpy as np
# import tensorflow as tf
# from google.cloud import bigquery
# from dotenv import load_dotenv

# load_dotenv()

# def get_live_weather_forecast():
#     """Fetches real 72-hour forecast for Karachi from Open-Meteo"""
#     print("☁️ Fetching live weather forecast from Open-Meteo...")
#     url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,wind_speed_10m"
#     res = requests.get(url).json()
#     return res['hourly']['temperature_2m'][:72], res['hourly']['wind_speed_10m'][:72]

# def run_inference():
#     # 1. Get the absolute path to the folder where THIS script lives (scripts/)
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Go up one level to the project root (Pearls_AQI_Predictor/)
#     project_root = os.path.dirname(script_dir)
    
#     # Create the perfect paths to your models
#     model_path = os.path.join(project_root, "models", "best_72h_model_ann.keras")
#     scaler_path = os.path.join(project_root, "models", "scaler.joblib")

#     # Load them (with error handling to tell us exactly what's wrong)
#     if not os.path.exists(model_path):
#         print(f"❌ ERROR: Model not found at {model_path}")
#         return
    
#     print(f"✅ Loading model from: {model_path}")
#     model = tf.keras.models.load_model(model_path)
#     scaler = joblib.load(scaler_path)

#     # 2. Get Current Air Quality from BigQuery
#     client = bigquery.Client.from_service_account_json(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
#     query = f"""
#         SELECT aqi, temp, humidity, wind_speed 
#         FROM `{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}` 
#         ORDER BY timestamp DESC LIMIT 2
#     """
#     latest_data = client.query(query).to_dataframe()
    
#     curr_aqi = latest_data.iloc[0]['aqi']
#     prev_aqi = latest_data.iloc[1]['aqi']
    
#     # 3. Get the 72h Weather Forecast
#     f_temps, f_winds = get_live_weather_forecast()

#     # 4. Prepare Feature Row (Must match the training column order exactly!)
#     # Base features
#     input_data = {
#         'temp': [latest_data.iloc[0]['temp']],
#         'humidity': [latest_data.iloc[0]['humidity']],
#         'wind_speed': [latest_data.iloc[0]['wind_speed']],
#         'hour': [pd.Timestamp.now().hour],
#         'day_of_week': [pd.Timestamp.now().dayofweek],
#         'aqi_lag_1h': [curr_aqi],
#         'aqi_change_rate': [(curr_aqi - prev_aqi) / (prev_aqi + 1e-6)]
#     }

#     # Weather leads (the 72h forecast)
#     for i in range(72):
#         input_data[f'temp_f_{i+1}h'] = [f_temps[i]]
#         input_data[f'wind_f_{i+1}h'] = [f_winds[i]]

#     input_df = pd.DataFrame(input_data)
    
#     # 5. Scale and Predict
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled, verbose=0)[0]

#     # 6. Display the result
#     print("\n" + "="*30)
#     print("🚀 KARACHI 72-HOUR AQI FORECAST")
#     print("="*30)
#     now = pd.Timestamp.now()
#     for i, val in enumerate(prediction):
#         # Only print every 6 hours to keep it clean, or print all
#         forecast_time = (now + pd.Timedelta(hours=i+1)).strftime('%Y-%m-%d %H:00')
#         print(f"{forecast_time} | Expected AQI: {val:.2f}")

# if __name__ == "__main__":
#     run_inference()


import os
import sys
import requests
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from dotenv import load_dotenv

# --- PATH FIX ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

def get_live_weather_forecast():
    """Fetches real 72-hour forecast for Karachi from Open-Meteo"""
    print("☁️ Fetching live weather forecast from Open-Meteo...")
    url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,wind_speed_10m"
    res = requests.get(url).json()
    return res['hourly']['temperature_2m'][:72], res['hourly']['wind_speed_10m'][:72]

def run_inference():
    # 1. Paths and Model Loading
    model_path = os.path.join(project_root, "models", "best_72h_model_ann.keras")
    scaler_path = os.path.join(project_root, "models", "scaler.joblib")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Get Data from MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    cursor = collection.find({"city": "Karachi"}).sort("timestamp", -1).limit(2)
    latest_data_list = list(cursor)
    
    latest_row = latest_data_list[0]
    prev_row = latest_data_list[1]
    
    # 3. Get Weather Forecast
    f_temps, f_winds = get_live_weather_forecast()

    # 4. Prepare Features
    input_data = {
        'temp': latest_row['temp'],
        'humidity': latest_row['humidity'],
        'wind_speed': latest_row['wind_speed'],
        'hour': pd.Timestamp.now().hour,
        'day_of_week': pd.Timestamp.now().dayofweek,
        'aqi_lag_1h': latest_row['aqi'],
        'aqi_change_rate': (latest_row['aqi'] - prev_row['aqi']) / (prev_row['aqi'] + 1e-6)
    }

    for i in range(72):
        input_data[f'temp_f_{i+1}h'] = f_temps[i]
        input_data[f'wind_f_{i+1}h'] = f_winds[i]

    input_df = pd.DataFrame([input_data])
    
    # Re-order features to match training
    if hasattr(scaler, 'feature_names_in_'):
        input_df = input_df[scaler.feature_names_in_]
    
    # 5. Predict Hourly
    input_scaled = scaler.transform(input_df)
    hourly_preds = model.predict(input_scaled, verbose=0)[0]

    # --- 6. NEW: GROUP BY DAY ---
    print("\n" + "="*45)
    print("🚀 KARACHI 3-DAY AQI FORECAST (Daily Average)")
    print("="*45)
    
    now = pd.Timestamp.now()
    
    # Split 72 hours into 3 chunks of 24 hours
    day_1 = hourly_preds[0:24]
    day_2 = hourly_preds[24:48]
    day_3 = hourly_preds[48:72]
    
    daily_summaries = [
        ("Tomorrow", day_1.mean(), now + pd.Timedelta(days=1)),
        ("Day After Tomorrow", day_2.mean(), now + pd.Timedelta(days=2)),
        ("Third Day", day_3.mean(), now + pd.Timedelta(days=3))
    ]

    for label, avg_aqi, date in daily_summaries:
        status = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Unhealthy"
        date_str = date.strftime('%Y-%m-%d')
        print(f"{label:20} ({date_str}) | Avg AQI: {avg_aqi:6.2f} | {status}")
    
    print("="*45)

    # --- 7. SAVE TO MONGODB (Fixed for NumPy types) ---
    print("📡 Saving predictions to MongoDB...")

    # Convert NumPy values to standard Python floats using .item()
    update_payload = {
        "timestamp": latest_row['timestamp'],  
        "actual_aqi_index": float(latest_row['aqi']), # Ensure this is a float
        "target_day_1_aqi": float(daily_summaries[0][1]), # .item() or float() works
        "target_day_2_aqi": float(daily_summaries[1][1]),
        "target_day_3_aqi": float(daily_summaries[2][1]),
        "prediction_run_at": pd.Timestamp.now()
    }

    try:
        # Update the record in MongoDB
        collection.update_one(
            {"_id": latest_row["_id"]}, 
            {"$set": update_payload},
            upsert=True
        )
        print("✅ Database updated successfully!")
    except Exception as e:
        print(f"❌ Failed to save to MongoDB: {e}")

if __name__ == "__main__":
    run_inference()