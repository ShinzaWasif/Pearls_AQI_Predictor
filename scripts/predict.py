
# import os
# import sys
# import requests
# import pandas as pd
# import joblib
# import numpy as np
# import tensorflow as tf
# from pymongo import MongoClient
# from dotenv import load_dotenv
# import mlflow
# import mlflow.keras

# # --- PATH FIX ---
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# load_dotenv(os.path.join(project_root, '.env'))

# def get_live_weather_forecast():
#     """Fetches real 72-hour forecast for Karachi from Open-Meteo"""
#     print("☁️ Fetching live weather forecast from Open-Meteo...")
#     url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,wind_speed_10m"
#     res = requests.get(url).json()
#     return res['hourly']['temperature_2m'][:72], res['hourly']['wind_speed_10m'][:72]

# def run_inference():
#     # 1. DagsHub / MLFLOW CONFIG
#     print("🚀 Connecting to DagsHub Model Registry...")
#     os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
#     os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
    
#     dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
#     mlflow.set_tracking_uri(dagshub_url)

#     # 2. Load Model and Scaler
#     try:
#         # Load the model directly from DagsHub Registry
#         model_name = "AQI_ANN_72h"
#         model_version = "latest" 
#         model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_version}")
#         print("✅ Model loaded successfully from DagsHub!")
        
#         # Scaler is usually kept locally or as an artifact
#         scaler_path = os.path.join(project_root, "models", "scaler.joblib")
#         if os.path.exists(scaler_path):
#             scaler = joblib.load(scaler_path)
#         else:
#             # Fallback: Try to download scaler from MLflow artifacts if you logged it there
#             print("⚠️ Local scaler not found, check your /models folder.")
#             return

#     except Exception as e:
#         print(f"❌ ERROR loading model/scaler: {e}")
#         return

#     # 3. Get Data from MongoDB
#     client = MongoClient(os.getenv("MONGO_URI"))
#     db = client[os.getenv("MONGO_DB_NAME")]
#     collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
#     cursor = collection.find({"city": "Karachi"}).sort("timestamp", -1).limit(2)
#     latest_data_list = list(cursor)
    
#     if len(latest_data_list) < 2:
#         print("❌ Not enough data in MongoDB to calculate lag features.")
#         return

#     latest_row = latest_data_list[0]
#     prev_row = latest_data_list[1]
    
#     # 4. Get Weather Forecast
#     f_temps, f_winds = get_live_weather_forecast()

#     # 5. Prepare Features
#     # 5. Prepare Features (MATCHING TRAINING FEATURES)
#     now = pd.Timestamp.now()
    
#     # Calculate some of the missing features
#     input_data = {
#         'temp': latest_row['temp'],
#         'humidity': latest_row['humidity'],
#         'wind_speed': latest_row['wind_speed'],
#         'hour': now.hour,
#         'day_of_week': now.dayofweek,
#         'month': now.month,                                  # Added 'month'
#         'is_winter': 1 if now.month in [11, 12, 1, 2] else 0, # Added 'is_winter'
#         'aqi': latest_row['aqi'],                            # Added 'aqi'
#         'aqi_lag_1h': latest_row['aqi'],
#         'aqi_change_rate': (latest_row['aqi'] - prev_row['aqi']) / (prev_row['aqi'] + 1e-6),
        
#         # For rolling/index columns that the scaler expects but aren't useful for current prediction:
#         'aqi_rolling_6h': latest_row['aqi'],                 # Fallback: use current aqi
#         'actual_aqi_index': 0,                               # Placeholders to satisfy the scaler
#         'target_day_1_aqi': 0,
#         'target_day_2_aqi': 0,
#         'target_day_3_aqi': 0
#     }

#     # Add the 72h forecast temperatures and wind speeds
#     for i in range(72):
#         input_data[f'temp_f_{i+1}h'] = f_temps[i]
#         input_data[f'wind_f_{i+1}h'] = f_winds[i]

#     input_df = pd.DataFrame([input_data])
    
#     # ⚠️ CRITICAL: Ensure all columns expected by the scaler exist
#     # Even if they are targets, the scaler was fitted on them, so they must be present
#     for col in scaler.feature_names_in_:
#         if col not in input_df.columns:
#             input_df[col] = 0  # Fill missing expected columns with 0
    
#     # Re-order features to match training expectations exactly
#     input_df = input_df[list(scaler.feature_names_in_)]
    
#     # 6. Predict Hourly
#     input_scaled = scaler.transform(input_df)
#     hourly_preds = model.predict(input_scaled, verbose=0)[0]

#     # --- 7. GROUP BY DAY ---
#     print("\n" + "="*45)
#     print("🚀 KARACHI 3-DAY AQI FORECAST (Daily Average)")
#     print("="*45)
    
#     now = pd.Timestamp.now()
#     day_1 = hourly_preds[0:24]
#     day_2 = hourly_preds[24:48]
#     day_3 = hourly_preds[48:72]
    
#     daily_summaries = [
#         ("Tomorrow", day_1.mean(), now + pd.Timedelta(days=1)),
#         ("Day After Tomorrow", day_2.mean(), now + pd.Timedelta(days=2)),
#         ("Third Day", day_3.mean(), now + pd.Timedelta(days=3))
#     ]

#     for label, avg_aqi, date in daily_summaries:
#         status = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Unhealthy"
#         date_str = date.strftime('%Y-%m-%d')
#         print(f"{label:20} ({date_str}) | Avg AQI: {avg_aqi:6.2f} | {status}")
    
#     # --- 8. SAVE TO MONGODB ---
#     print("📡 Saving predictions to MongoDB...")
#     update_payload = {
#         "timestamp": latest_row['timestamp'],  
#         "actual_aqi_index": float(latest_row['aqi']),
#         "target_day_1_aqi": float(daily_summaries[0][1]),
#         "target_day_2_aqi": float(daily_summaries[1][1]),
#         "target_day_3_aqi": float(daily_summaries[2][1]),
#         "prediction_run_at": pd.Timestamp.now()
#     }

#     try:
#         collection.update_one(
#             {"_id": latest_row["_id"]}, 
#             {"$set": update_payload},
#             upsert=True
#         )
#         print("✅ Database updated successfully!")
#     except Exception as e:
#         print(f"❌ Failed to save to MongoDB: {e}")

# if __name__ == "__main__":
#     run_inference()

# import os
# import sys
# import joblib
# import pandas as pd
# import numpy as np
# import requests
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from datetime import datetime

# # --- PATH FIX ---
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# load_dotenv(os.path.join(project_root, '.env'))

# def get_live_weather_forecast():
#     """Fetches real 72-hour forecast for Karachi from Open-Meteo"""
#     print("☁️ Fetching live weather forecast from Open-Meteo...")
#     # Updated to include relative humidity for the Smog Index
#     url = "https://api.open-meteo.com/v1/forecast?latitude=24.8607&longitude=67.0011&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
#     res = requests.get(url).json()
#     h = res['hourly']
#     return h['temperature_2m'][:72], h['wind_speed_10m'][:72], h['relative_humidity_2m'][:72]

# def run_inference():
#     # 1. Load Model and Scaler from local /models folder
#     # Using XGBoost as it was the champion in your training run
#     try:
#         models_dir = os.path.join(project_root, "models")
#         model = joblib.load(os.path.join(models_dir, "best_xgb.joblib"))
#         scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
#         print("✅ XGBoost Model and Scaler loaded locally!")
#     except Exception as e:
#         print(f"❌ ERROR loading models: {e}")
#         return

#     # 2. Get Data from MongoDB
#     client = MongoClient(os.getenv("MONGO_URI"))
#     db = client[os.getenv("MONGO_DB_NAME")]
#     collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
#     # We need the last few records to calculate rolling averages and lags
#     cursor = collection.find({"city": "Karachi"}).sort("timestamp", -1).limit(10)
#     history = list(cursor)
    
#     if len(history) < 6:
#         print("❌ Not enough historical data in MongoDB (need at least 6 hours).")
#         return

#     latest_row = history[0]
    
#     # 3. Get Weather Forecast for the next 72h
#     f_temps, f_winds, f_humids = get_live_weather_forecast()

#     # 4. Prepare Features (MUST MATCH feature_engineering.py EXACTLY)
#     now = datetime.now()
    
#     # Calculate Cyclic Time
#     hour = now.hour
#     hour_sin = np.sin(2 * np.pi * hour / 24)
#     hour_cos = np.cos(2 * np.pi * hour / 24)
    
#     # Calculate Smog Index
#     is_winter = 1 if now.month in [11, 12, 1, 2] else 0
#     smog_idx = (latest_row['humidity'] / (latest_row['wind_speed'] + 1)) * is_winter

#     # Calculate Lags and Rolling from MongoDB history
#     aqi_history = [h['aqi_calibrated'] for h in history]
#     aqi_lag_1h = aqi_history[0]
#     aqi_lag_2h = aqi_history[1]
#     aqi_rolling_6h = np.mean(aqi_history[:6])
#     aqi_change_rate = (aqi_lag_1h - aqi_lag_2h) / (aqi_lag_2h + 0.1)

#     # Construct feature dictionary
#     input_data = {
#         'aqi_calibrated': latest_row['aqi_calibrated'],
#         'temp': latest_row['temp'],
#         'humidity': latest_row['humidity'],
#         'wind_speed': latest_row['wind_speed'],
#         'hour_sin': hour_sin,
#         'hour_cos': hour_cos,
#         'is_winter': is_winter,
#         'smog_index': smog_idx,
#         'aqi_rolling_6h': aqi_rolling_6h,
#         'aqi_lag_1h': aqi_lag_1h,
#         'aqi_change_rate': aqi_change_rate
#     }

#     # Add 72h Forecast features (temp_f_1h, wind_f_1h, etc.)
#     for i in range(72):
#         input_data[f'temp_f_{i+1}h'] = f_temps[i]
#         input_data[f'wind_f_{i+1}h'] = f_winds[i]

#     # Convert to DataFrame and align with scaler
#     input_df = pd.DataFrame([input_data])
    
#     # Align columns with what the scaler expects (this fixes NameErrors)
#     expected_features = list(scaler.feature_names_in_)
#     for col in expected_features:
#         if col not in input_df.columns:
#             input_df[col] = 0
            
#     input_df = input_df[expected_features]

#     # 5. Predict
#     input_scaled = scaler.transform(input_df)
#     hourly_preds = model.predict(input_scaled)[0]

#     # 6. Display Results
#     print("\n" + "="*50)
#     print(f"🌍 KARACHI AQI FORECAST - Generated at {now.strftime('%H:%M')}")
#     print("="*50)
    
#     daily_avgs = [
#         ("Tomorrow", np.mean(hourly_preds[0:24])),
#         ("Day 2", np.mean(hourly_preds[24:48])),
#         ("Day 3", np.mean(hourly_preds[48:72]))
#     ]

#     for label, avg in daily_avgs:
#         # EPA Category Logic
#         status = "Good" if avg <= 50 else "Moderate" if avg <= 100 else "Unhealthy"
#         if avg > 150: status = "VERY Unhealthy"
#         print(f"📅 {label:12} | Predicted Avg AQI: {avg:6.2f} | [{status}]")

#     # 7. Save prediction back to the most recent record
#     # This allows your dashboard to show "Actual vs Predicted"
#     update_payload = {
#         "is_predicted": True,
#         "predicted_72h": hourly_preds.tolist(),
#         "prediction_run_at": datetime.now()
#     }

#     try:
#         collection.update_one({"_id": latest_row["_id"]}, {"$set": update_payload})
#         print("\n✅ Predictions saved to MongoDB for latest record.")
#     except Exception as e:
#         print(f"❌ Database update failed: {e}")

# if __name__ == "__main__":
#     run_inference()

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
    # 1. LOAD MODEL FROM REGISTRY
    try:
        model_name = "AQI_72h_Karachi"
        model_uri = f"models:/{model_name}/latest"
        
        print(f"📡 Connecting to DagsHub Registry: {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        models_dir = os.path.join(project_root, "models")
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        print("✅ Registry Model and Local Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR connecting to Model Registry: {e}")
        return

    # 2. Get the PRE-COMPUTED row from MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    # We grab the latest row that was processed by the Feature Pipeline
    # It already contains the weather forecast, rolling averages, and smog index!
    latest_row = collection.find_one({"city": "Karachi"}, sort=[("timestamp", -1)])
    
    if not latest_row:
        print("❌ No data found in MongoDB.")
        return

    # 3. Prepare Features (Using Pre-computed data)
    # We only need to align the columns to what the scaler expects
    target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    drop_cols = ['_id', 'timestamp', 'city', 'aqi', 'aqi_calibrated', 'is_predicted', 'predicted_72h'] + target_cols
    
    input_df = pd.DataFrame([latest_row])
    
    # Ensure we only have numeric features that were used in training
    expected_features = list(scaler.feature_names_in_)
    # Temporary debug lines
    print(f"DEBUG: Scaler expects {len(expected_features)} features.")
    print(f"DEBUG: Features are: {expected_features}")
    
    # Fill any missing columns with 0 and filter for model features
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_features]

    # 4. PREDICT
    input_scaled = scaler.transform(input_df)
    hourly_preds = model.predict(input_scaled)
    
    if len(hourly_preds.shape) > 1:
        hourly_preds = hourly_preds[0]

    # Ensure no negative predictions (AQI can't be below 0)
    hourly_preds = np.maximum(hourly_preds, 0)

    # 5. DISPLAY & SAVE
    print("\n" + "="*50)
    print(f"🌍 KARACHI LIVE FORECAST | Time: {latest_row['timestamp']}")
    print("="*50)
    
    daily_avgs = [
        ("Next 24h", np.mean(hourly_preds[0:24])),
        ("24h - 48h", np.mean(hourly_preds[24:48])),
        ("48h - 72h", np.mean(hourly_preds[48:72]))
    ]

    for label, avg in daily_avgs:
        status = "Good" if avg <= 50 else "Moderate" if avg <= 100 else "Unhealthy"
        if avg > 150: status = "Hazardous"
        print(f"📅 {label:12} | Avg AQI: {avg:6.2f} | [{status}]")

    # Update the EXACT row in MongoDB with the results
    update_payload = {
        "is_predicted": True,
        "predicted_72h": hourly_preds.tolist(),
        "prediction_run_at": datetime.now(),
        "registry_model_version": model_uri
    }

    collection.update_one({"_id": latest_row["_id"]}, {"$set": update_payload})
    print(f"\n✅ Prediction cycle complete for {latest_row['timestamp']}")

if __name__ == "__main__":
    run_inference()