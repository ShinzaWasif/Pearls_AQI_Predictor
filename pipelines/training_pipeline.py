
# import os
# import joblib
# import pandas as pd
# import tensorflow as tf
# from pymongo import MongoClient
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.linear_model import Ridge
# from xgboost import XGBRegressor
# from dotenv import load_dotenv
# import mlflow
# import mlflow.keras
# import mlflow.sklearn

# load_dotenv()

# # --- DAGSHUB / MLFLOW CONFIG ---
# # Ensure these are in your .env: DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
# os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
# os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
# dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
# mlflow.set_tracking_uri(dagshub_url)
# mlflow.set_experiment("AQI_72h_Forecasting")

# def build_ann_model(input_dim):
#     """Builds the Neural Network architecture for 72-hour output"""
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(72) # Output layer: 72 neurons for 72 hours
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def run_training():
#     # --- 1. Load Data from MongoDB ---
#     print(f"🔌 Connecting to MongoDB for training data...")
#     client = MongoClient(os.getenv("MONGO_URI"))
#     db = client[os.getenv("MONGO_DB_NAME")]
#     collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
#     # Fetch data and convert to DataFrame
#     cursor = collection.find({"city": "Karachi"})
#     df = pd.DataFrame(list(cursor))
    
#     if df.empty:
#         print("❌ Error: No training data found in MongoDB. Run backfill first!")
#         return

#     # Drop MongoDB internal ID
#     if '_id' in df.columns:
#         df = df.drop(columns=['_id'])

#     # --- 2. Identify Columns ---
#     target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    
#     # Filter for only numeric columns to avoid the Timestamp error
#     # This automatically ignores 'timestamp', 'city', and 'prediction_run_at'
#     X = df.drop(columns=target_cols, errors='ignore').select_dtypes(include=['number'])
#     y = df[target_cols]
#     # --- New Cleaning Step ---
#     print("🧹 Cleaning missing values (NaNs)...")
    
#     # Fill NaNs in features with the mean of each column
#     X = X.fillna(X.mean())
    
#     # Fill NaNs in targets (y) as well, or drop those rows
#     y = y.fillna(y.mean())
    
#     # Double check: if any NaNs remain (like in a column that was all NaNs)
#     X = X.fillna(0)
#     y = y.fillna(0)
#     print(f"📊 Training with {X.shape[1]} numeric features.")
#     print(f"📋 Columns used: {list(X.columns)}")

#     # --- 3. Scale and Split ---
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # --- 4. Train and Log with MLflow ---
#     with mlflow.start_run(run_name="MultiModel_Training"):
#         # Log basic info
#         mlflow.log_param("data_points", len(df))
#         mlflow.log_param("features_count", X_train.shape[1])

#         # A. XGBoost
#         print(f"🌲 Training Multi-Output XGBoost...")
#         xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
#         xgb.fit(X_train, y_train)
        
#         # B. Ridge
#         print("📈 Training Multi-Output Ridge...")
#         ridge = Ridge(alpha=1.0)
#         ridge.fit(X_train, y_train)

#         # C. ANN
#         print("🧠 Training TensorFlow ANN (50 epochs)...")
#         ann = build_ann_model(X_train.shape[1])
#         ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

#         # --- 5. Evaluate and Log Metrics ---
#         models = {"XGBoost": xgb, "Ridge": ridge, "TensorFlow": ann}
#         for name, m in models.items():
#             preds = m.predict(X_test)
#             mae = mean_absolute_error(y_test, preds)
#             r2 = r2_score(y_test, preds)
#             print(f"📌 {name}: Avg R² = {r2:.4f}, Avg MAE = {mae:.4f}")
            
#             # Log metrics to DagsHub
#             mlflow.log_metric(f"{name}_R2", r2)
#             mlflow.log_metric(f"{name}_MAE", mae)

#         # --- 6. Save and Register Models ---
#         project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         models_dir = os.path.join(project_root, "models")
#         os.makedirs(models_dir, exist_ok=True)

#         # Local saves
#         joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
#         joblib.dump(xgb, os.path.join(models_dir, "best_72h_model_xgb.joblib"))
#         ann.save(os.path.join(models_dir, "best_72h_model_ann.keras"))

#         # MLflow/DagsHub Model Registry
#         mlflow.sklearn.log_model(xgb, "xgb_model", registered_model_name="AQI_XGBoost_72h")
#         mlflow.keras.log_model(ann, "ann_model", registered_model_name="AQI_ANN_72h")
#         mlflow.log_artifact(os.path.join(models_dir, "scaler.joblib"))

#         print(f"✅ Training Complete. Models registered on DagsHub!")

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
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import warnings

# Ignore FutureWarnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# --- MLFLOW / DAGSHUB CONFIG ---
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("AQI_72h_Forecasting_v2")

def build_ann_model(input_dim):
    """Neural Network optimized for 72-hour non-linear AQI prediction."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(72) # Predicting 72 hourly points
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def run_training():
    # 1. Fetching Latest Calibrated Data
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    cursor = collection.find({"city": "Karachi"})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("❌ Error: MongoDB is empty. Run your feature pipeline first.")
        return

    # 2. Advanced Feature Selection
    target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    drop_cols = ['_id', 'timestamp', 'city', 'aqi'] + target_cols
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=['number'])
    y = df[target_cols]

    # Data Cleaning
    X = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
    y = y.fillna(y.median())

    print(f"📊 Training on {X.shape[1]} features (Smog Index, Hour_Sin, etc.)")

    # 3. Scaling & Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    with mlflow.start_run(run_name=f"Karachi_Final_{datetime.now().strftime('%m%d')}"):
        mlflow.log_params({"features_count": X.shape[1], "model_type": "Multi-Model Ensemble"})

        # A. Multi-Output XGBoost
        print("🌲 Training XGBoost...")
        xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05))
        xgb_model.fit(X_train, y_train)

        # B. TensorFlow ANN
        print("🧠 Training Neural Network...")
        ann_model = build_ann_model(X_train.shape[1])
        ann_model.fit(X_train, y_train, epochs=60, batch_size=32, verbose=0, validation_split=0.1)

        # 4. Evaluation
        results = {}
        for name, model in {"XGBoost": xgb_model, "ANN": ann_model}.items():
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            mlflow.log_metric(f"{name}_MAE", mae)
            mlflow.log_metric(f"{name}_R2", r2)
            results[name] = mae
            print(f"📌 {name} -> MAE: {mae:.2f}, R2: {r2:.2f}")

        # 5. Save and Register Models
        # This fixes the 'project_root' NameError
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # Goes up from 'pipelines/' to project root
        
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save Scaler and local model copies
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(xgb_model, os.path.join(models_dir, "best_xgb.joblib"))
        ann_model.save(os.path.join(models_dir, "best_ann.keras"))
        
        mlflow.log_artifact(os.path.join(models_dir, "scaler.joblib"))

        # Register the winner to DagsHub
        best_model_name = min(results, key=results.get)
        print(f"🏆 Best Model based on MAE: {best_model_name}")

        if best_model_name == "XGBoost":
            mlflow.sklearn.log_model(xgb_model, "best_model", registered_model_name="AQI_72h_Karachi")
        else:
            mlflow.keras.log_model(ann_model, "best_model", registered_model_name="AQI_72h_Karachi")

        print(f"✅ Training Complete. Best model saved in {models_dir} and registered on DagsHub!")

if __name__ == "__main__":
    run_training()