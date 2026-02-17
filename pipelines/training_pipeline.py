# import os
# import joblib
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from pymongo import MongoClient
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import (
#     mean_absolute_error, r2_score, mean_squared_error, 
#     mean_absolute_percentage_error, median_absolute_error
# )
# from xgboost import XGBRegressor
# from dotenv import load_dotenv
# import mlflow
# import mlflow.keras
# import mlflow.sklearn
# from datetime import datetime
# import warnings

# # --- CONFIG & SUPPRESSION ---
# warnings.filterwarnings("ignore", category=FutureWarning)
# load_dotenv()

# # MLFLOW / DAGSHUB CONFIG
# os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
# os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
# dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
# mlflow.set_tracking_uri(dagshub_url)
# mlflow.set_experiment("AQI_72h_Forecasting_v2")

# # --- MODEL BUILDERS ---
# def build_ann_model(input_dim):
#     """Deep Neural Network (ANN) for non-linear regression."""
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(72)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def build_lstm_model(input_dim):
#     """LSTM model to capture temporal sequences in the data."""
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Reshape((1, input_dim)), 
#         tf.keras.layers.LSTM(128, return_sequences=True),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(72) 
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def run_training():
#     # 1. Fetching Data
#     client = MongoClient(os.getenv("MONGO_URI"))
#     db = client[os.getenv("MONGO_DB_NAME")]
#     collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
#     cursor = collection.find({"city": "Karachi"})
#     df = pd.DataFrame(list(cursor))
    
#     if df.empty:
#         print("❌ Error: MongoDB is empty. Run your feature pipeline first.")
#         return

#     # 2. Feature & Target Selection
#     target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
#     drop_cols = ['_id', 'timestamp', 'city', 'aqi'] + target_cols
    
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=['number'])
#     y = df[target_cols]

#     X = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
#     y = y.fillna(y.median())

#     print(f"📊 Training on {X.shape[1]} features for 72-hour forecast.")

#     # 3. Scaling & Split
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

#     with mlflow.start_run(run_name=f"Karachi_Ensemble_{datetime.now().strftime('%m%d_%H%M')}"):
#         mlflow.log_params({"features_count": X.shape[1], "ensemble_size": 3})

#         # --- MODEL A: XGBoost ---
#         print("🌲 Training XGBoost...")
#         xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05))
#         xgb_model.fit(X_train, y_train)

#         # --- MODEL B: ANN ---
#         print("🧠 Training Neural Network...")
#         ann_model = build_ann_model(X_train.shape[1])
#         ann_model.fit(X_train, y_train, epochs=60, batch_size=32, verbose=0, validation_split=0.1)

#         # --- MODEL C: LSTM ---
#         print("⏳ Training LSTM...")
#         lstm_model = build_lstm_model(X_train.shape[1])
#         lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)

#         # 4. ENHANCED EVALUATION & DB LOGGING
#         results_mae = {}
#         full_metrics_for_db = {}
#         model_objs = {"XGBoost": xgb_model, "ANN": ann_model, "LSTM": lstm_model}

#         print("\n" + "="*50)
#         print("🚀 KARACHI AQI MODEL PERFORMANCE REPORT")
#         print("="*50)

#         for name, model in model_objs.items():
#             preds = model.predict(X_test)
            
#             # --- CALCULATE 5 METRICS ---
#             mae = mean_absolute_error(y_test, preds)
#             # RMSE for multi-output: average of per-output RMSEs
#             rmse = np.mean(np.sqrt(mean_squared_error(y_test, preds, multioutput='raw_values')))
#             r2 = r2_score(y_test, preds)
#             mape = mean_absolute_percentage_error(y_test, preds)
#             medae = median_absolute_error(y_test, preds)
            
#             # --- LOG TO MLFLOW ---
#             mlflow.log_metric(f"{name}_MAE", mae)
#             mlflow.log_metric(f"{name}_RMSE", rmse)
#             mlflow.log_metric(f"{name}_R2", r2)
#             mlflow.log_metric(f"{name}_MAPE", mape)
#             mlflow.log_metric(f"{name}_MedAE", medae)
            
#             # Store for Database & Selection
#             results_mae[name] = mae
#             full_metrics_for_db[name] = {
#                 "MAE": float(mae),
#                 "RMSE": float(rmse),
#                 "R2": float(r2),
#                 "MAPE": float(mape),
#                 "MedAE": float(medae)
#             }
            
#             # --- TERMINAL REPORT ---
#             print(f"📊 MODEL: {name}")
#             print(f"   🔹 MAE:   {mae:.2f} (Avg error)")
#             print(f"   🔹 RMSE:  {rmse:.2f} (Penalty for outliers)")
#             print(f"   🔹 MAPE:  {mape*100:.2f}% (Percent error)")
#             print(f"   🔹 R2:    {r2:.2f}")
#             print(f"   🔹 MedAE: {medae:.2f} (Robust error)")
#             print("-" * 30)

#         print("="*50)

#         # 5. LOCAL SAVING
#         models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
#         os.makedirs(models_dir, exist_ok=True)
#         joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
#         joblib.dump(xgb_model, os.path.join(models_dir, "best_xgb.joblib"))
#         ann_model.save(os.path.join(models_dir, "best_ann.keras"))
#         lstm_model.save(os.path.join(models_dir, "best_lstm.keras"))
#         mlflow.log_artifact(os.path.join(models_dir, "scaler.joblib"))

#         # 6. CHAMPION SELECTION & REGISTRY
#         best_model_name = min(results_mae, key=results_mae.get)
#         print(f"🏆 Champion Model: {best_model_name}")

#         if best_model_name == "XGBoost":
#             mlflow.sklearn.log_model(xgb_model, "best_model", registered_model_name="AQI_72h_Karachi")
#         else: # Deep Learning models (ANN or LSTM)
#             mlflow.keras.log_model(model_objs[best_model_name], "best_model", registered_model_name="AQI_72h_Karachi")

#         # 7. SAVE ALL METRICS TO MONGODB
#         print("💾 Saving training metadata to MongoDB...")
#         performance_audit = {
#             "timestamp": datetime.now(),
#             "experiment": "AQI_72h_Forecasting_v2",
#             "champion_model": best_model_name,
#             "mlflow_run_id": mlflow.active_run().info.run_id,
#             "metrics": full_metrics_for_db
#         }
#         db["model_performance_history"].insert_one(performance_audit)

#         print(f"✅ Registered on DagsHub & Logged to MongoDB!")

# if __name__ == "__main__":
#     run_training()

import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    mean_absolute_percentage_error, median_absolute_error
)
from xgboost import XGBRegressor
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import mlflow.sklearn
from datetime import datetime
import warnings

# --- CONFIG & SUPPRESSION ---
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# MLFLOW / DAGSHUB CONFIG
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("AQI_72h_Forecasting_v2")

# --- MODEL BUILDERS ---
def build_ann_model(input_dim):
    """Deep Neural Network (ANN) for non-linear regression."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(72)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm_model(input_dim):
    """LSTM model to capture temporal sequences in the data."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((1, input_dim)), 
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(72) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def run_training():
    # 1. Fetching Data
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    # We fetch data that has the 72h targets (generated during backfill/feature pipeline)
    cursor = collection.find({"city": "Karachi", "target_aqi_72h": {"$exists": True}})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("❌ Error: No valid training data found in MongoDB. Please run the Backfill Pipeline first.")
        return

    # 2. Feature & Target Selection
    # CRITICAL: We drop 'aqi' (raw) and keep 'aqi_calibrated' (multiplied by 1.42)
    target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    
    # We explicitly drop the raw 'aqi' to prevent the model from learning from unscaled data
    drop_cols = ['_id', 'timestamp', 'city', 'aqi'] + target_cols
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=['number'])
    y = df[target_cols]

    # Handle missing values using median (robust for Karachi AQI spikes)
    X = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
    y = y.fillna(y.median())

    print(f"📊 Training on {X.shape[1]} features (including Calibrated AQI) for 72-hour forecast.")

    # 3. Scaling & Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    with mlflow.start_run(run_name=f"Karachi_Calibrated_Ensemble_{datetime.now().strftime('%m%d_%H%M')}"):
        mlflow.log_params({"features_count": X.shape[1], "ensemble_size": 3, "calibration_factor": 1.42})

        # --- MODEL A: XGBoost ---
        print("🌲 Training XGBoost...")
        xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05))
        xgb_model.fit(X_train, y_train)

        # --- MODEL B: ANN ---
        print("🧠 Training Neural Network...")
        ann_model = build_ann_model(X_train.shape[1])
        ann_model.fit(X_train, y_train, epochs=60, batch_size=32, verbose=0, validation_split=0.1)

        # --- MODEL C: LSTM ---
        print("⏳ Training LSTM...")
        lstm_model = build_lstm_model(X_train.shape[1])
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)

        # 4. ENHANCED EVALUATION
        results_mae = {}
        full_metrics_for_db = {}
        model_objs = {"XGBoost": xgb_model, "ANN": ann_model, "LSTM": lstm_model}

        print("\n" + "="*50)
        print("🚀 KARACHI AQI MODEL PERFORMANCE (CALIBRATED)")
        print("="*50)

        for name, model in model_objs.items():
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.mean(np.sqrt(mean_squared_error(y_test, preds, multioutput='raw_values')))
            r2 = r2_score(y_test, preds)
            mape = mean_absolute_percentage_error(y_test, preds)
            
            mlflow.log_metric(f"{name}_MAE", mae)
            mlflow.log_metric(f"{name}_R2", r2)
            mlflow.log_metric(f"{name}_RMSE", rmse)
            mlflow.log_metric(f"{name}_MAPE", mape)

            
            results_mae[name] = mae
            full_metrics_for_db[name] = {
                "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": float(mape)
            }
            
            print(f"📊 MODEL: {name}\n   🔹 MAE: {mae:.2f} | 🔹 R2: {r2:.2f} ")

        # 5. REGISTRY LOGGING (SCALER & CHAMPION)
        # We save the scaler so the Prediction Script can use the exact same normalization
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        best_model_name = min(results_mae, key=results_mae.get)
        print(f"🏆 Champion Model: {best_model_name}")

        if best_model_name == "XGBoost":
            mlflow.sklearn.log_model(xgb_model, "best_model", registered_model_name="AQI_72h_Karachi_Calibrated")
        else:
            mlflow.keras.log_model(model_objs[best_model_name], "best_model", registered_model_name="AQI_72h_Karachi_Calibrated")

        # 6. SAVE PERFORMANCE HISTORY
        performance_audit = {
            "timestamp": datetime.now(),
            "champion_model": best_model_name,
            "calibration_used": 1.42,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "metrics": full_metrics_for_db
        }
        db["model_performance_history"].insert_one(performance_audit)

        # Cleanup local scaler file after logging
        if os.path.exists(scaler_path):
            os.remove(scaler_path)

        print(f"✅ Training Complete. Best model ({best_model_name}) is now live!")

if __name__ == "__main__":
    run_training()