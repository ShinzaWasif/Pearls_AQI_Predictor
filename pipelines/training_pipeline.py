
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import joblib
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from google.cloud import bigquery
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, r2_score
# from dotenv import load_dotenv

# # Models
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge  # Requested
# from xgboost import XGBRegressor

# load_dotenv()

# def build_tensorflow_model(input_shape):
#     model = tf.keras.Sequential([
#         # This is the "Modern" way Keras wants it:
#         tf.keras.layers.Input(shape=(input_shape,)), 
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# def run_multi_model_training():
#     # 1. Fetch Data
#     client = bigquery.Client.from_service_account_json(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
#     table_id = f"{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
#     df = client.query(f"SELECT * FROM `{table_id}` WHERE city = 'Karachi'").to_dataframe()

#     X = df.drop(columns=['aqi', 'timestamp', 'city'])
#     y = df['aqi']
    
#     # 2. Feature Scaling (Crucial for Ridge & TensorFlow)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # 3. Define Models
#     results = []
    
#     # --- Part A: Scikit-Learn & XGBoost ---
#     sk_models = {
#         "Ridge Regression": Ridge(alpha=1.0),
#         "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#     }

#     for name, model in sk_models.items():
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         mae = mean_absolute_error(y_test, preds)
#         r2 = r2_score(y_test, preds)
#         results.append({"Model": name, "MAE": mae, "R2": r2})
#         print(f"📌 {name}: R² = {r2:.4f}, MAE = {mae:.4f}")

#     # --- Part B: TensorFlow (Neural Network) ---
#     print("🧠 Training TensorFlow Neural Network...")
#     tf_model = build_tensorflow_model(X_train.shape[1])
#     # verbose=0 keeps the console clean
#     tf_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0) 
    
#     tf_preds = tf_model.predict(X_test).flatten()
#     tf_mae = mean_absolute_error(y_test, tf_preds)
#     tf_r2 = r2_score(y_test, tf_preds)
#     results.append({"Model": "TensorFlow (ANN)", "MAE": tf_mae, "R2": tf_r2})
#     print(f"📌 TensorFlow (ANN): R² = {tf_r2:.4f}, MAE = {tf_mae:.4f}")

#     # 4. Show Comparison Table
#     print("-" * 50)
#     results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
#     print(results_df)

#     # 5. Save the Best Scikit-model and the Scaler
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(scaler, "models/scaler.joblib")
    
#     # Logic to save the winning model (simplified)
#     # Note: TF models are saved via model.save(), others via joblib.
#     print("-" * 50)
#     print("✅ Experiment Complete. Scaler and model metrics saved.")

# if __name__ == "__main__":
#     run_multi_model_training()

# import os
# import joblib
# import pandas as pd
# import tensorflow as tf
# from google.cloud import bigquery
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.linear_model import Ridge
# from xgboost import XGBRegressor
# from dotenv import load_dotenv

# load_dotenv()

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
#     # --- 1. Load Data ---
#     # Resolve absolute path for service account
#     creds_path = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
#     if not os.path.isabs(creds_path):
#         project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         creds_path = os.path.join(project_root, creds_path)

#     client = bigquery.Client.from_service_account_json(creds_path)
#     table_id = f"{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
    
#     print(f"📥 Fetching training data from {table_id}...")
#     df = client.query(f"SELECT * FROM `{table_id}` WHERE city = 'Karachi'").to_dataframe()

#     # --- 2. Identify Columns ---
#     target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
#     # Features exclude targets, timestamp, city metadata, and the raw aqi
#     feature_cols = [col for col in df.columns if col not in target_cols + ['timestamp', 'city', 'aqi']]
    
#     X = df[feature_cols]
#     y = df[target_cols]

#     # --- 3. Scale and Split ---
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # --- 4. Train Models ---
#     print(f"🌲 Training Multi-Output XGBoost on {X_train.shape[1]} features...")
#     xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
#     xgb.fit(X_train, y_train)
    
#     print("📈 Training Multi-Output Ridge...")
#     ridge = Ridge(alpha=1.0)
#     ridge.fit(X_train, y_train)

#     print("🧠 Training TensorFlow ANN (50 epochs)...")
#     ann = build_ann_model(X_train.shape[1])
#     ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

#     # --- 5. Evaluate ---
#     models = {"XGBoost": xgb, "Ridge": ridge, "TensorFlow": ann}
#     for name, m in models.items():
#         preds = m.predict(X_test)
#         mae = mean_absolute_error(y_test, preds)
#         r2 = r2_score(y_test, preds)
#         print(f"📌 {name}: Avg R² = {r2:.4f}, Avg MAE = {mae:.4f}")

#     # --- 6. Save Everything with Correct Formats ---
#     # We save to the root 'models' folder
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     models_dir = os.path.join(project_root, "models")
#     os.makedirs(models_dir, exist_ok=True)

#     # 1. Scaler (joblib)
#     joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    
#     # 2. XGBoost (joblib)
#     joblib.dump(xgb, os.path.join(models_dir, "best_72h_model_xgb.joblib"))
    
#     # 3. TensorFlow (MUST be .keras)
#     # This prevents the 'ValueError: File format not supported' error
#     ann.save(os.path.join(models_dir, "best_72h_model_ann.keras"))

#     print(f"✅ Training Complete. Models saved in: {models_dir}")

# if __name__ == "__main__":
#     run_training()

import os
import joblib
import pandas as pd
import tensorflow as tf
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import mlflow.sklearn

load_dotenv()

# --- DAGSHUB / MLFLOW CONFIG ---
# Ensure these are in your .env: DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
dagshub_url = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("AQI_72h_Forecasting")

def build_ann_model(input_dim):
    """Builds the Neural Network architecture for 72-hour output"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(72) # Output layer: 72 neurons for 72 hours
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def run_training():
    # --- 1. Load Data from MongoDB ---
    print(f"🔌 Connecting to MongoDB for training data...")
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    
    # Fetch data and convert to DataFrame
    cursor = collection.find({"city": "Karachi"})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("❌ Error: No training data found in MongoDB. Run backfill first!")
        return

    # Drop MongoDB internal ID
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # --- 2. Identify Columns ---
    target_cols = [f'target_aqi_{i}h' for i in range(1, 73)]
    
    # Filter for only numeric columns to avoid the Timestamp error
    # This automatically ignores 'timestamp', 'city', and 'prediction_run_at'
    X = df.drop(columns=target_cols, errors='ignore').select_dtypes(include=['number'])
    y = df[target_cols]
    # --- New Cleaning Step ---
    print("🧹 Cleaning missing values (NaNs)...")
    
    # Fill NaNs in features with the mean of each column
    X = X.fillna(X.mean())
    
    # Fill NaNs in targets (y) as well, or drop those rows
    y = y.fillna(y.mean())
    
    # Double check: if any NaNs remain (like in a column that was all NaNs)
    X = X.fillna(0)
    y = y.fillna(0)
    print(f"📊 Training with {X.shape[1]} numeric features.")
    print(f"📋 Columns used: {list(X.columns)}")

    # --- 3. Scale and Split ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- 4. Train and Log with MLflow ---
    with mlflow.start_run(run_name="MultiModel_Training"):
        # Log basic info
        mlflow.log_param("data_points", len(df))
        mlflow.log_param("features_count", X_train.shape[1])

        # A. XGBoost
        print(f"🌲 Training Multi-Output XGBoost...")
        xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
        xgb.fit(X_train, y_train)
        
        # B. Ridge
        print("📈 Training Multi-Output Ridge...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        # C. ANN
        print("🧠 Training TensorFlow ANN (50 epochs)...")
        ann = build_ann_model(X_train.shape[1])
        ann.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # --- 5. Evaluate and Log Metrics ---
        models = {"XGBoost": xgb, "Ridge": ridge, "TensorFlow": ann}
        for name, m in models.items():
            preds = m.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            print(f"📌 {name}: Avg R² = {r2:.4f}, Avg MAE = {mae:.4f}")
            
            # Log metrics to DagsHub
            mlflow.log_metric(f"{name}_R2", r2)
            mlflow.log_metric(f"{name}_MAE", mae)

        # --- 6. Save and Register Models ---
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Local saves
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(xgb, os.path.join(models_dir, "best_72h_model_xgb.joblib"))
        ann.save(os.path.join(models_dir, "best_72h_model_ann.keras"))

        # MLflow/DagsHub Model Registry
        mlflow.sklearn.log_model(xgb, "xgb_model", registered_model_name="AQI_XGBoost_72h")
        mlflow.keras.log_model(ann, "ann_model", registered_model_name="AQI_ANN_72h")
        mlflow.log_artifact(os.path.join(models_dir, "scaler.joblib"))

        print(f"✅ Training Complete. Models registered on DagsHub!")

if __name__ == "__main__":
    run_training()