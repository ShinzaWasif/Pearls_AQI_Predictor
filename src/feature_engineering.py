import pandas as pd
import numpy as np

# def compute_features(df):
#     # Ensure chronological order
#     df = df.sort_values('timestamp').reset_index(drop=True)

#     # 1. Basic Time Features
#     df['hour'] = df['timestamp'].dt.hour
#     df['day_of_week'] = df['timestamp'].dt.dayofweek
    
#     # 2. Historical Lags (Immediate Past)
#     df['aqi_lag_1h'] = df['aqi'].shift(1)
#     df['aqi_lag_2h'] = df['aqi'].shift(2)
#     df['aqi_change_rate'] = (df['aqi_lag_1h'] - df['aqi_lag_2h']) / (df['aqi_lag_2h'] + 1e-6)

#     # 3. Create a dictionary to hold all new "Future" columns
#     # This avoids the "Fragmentation" warning
#     future_data = {}

#     # --- Create Weather Forecast Features (Leads) ---
#     for i in range(1, 73):
#         future_data[f'temp_f_{i}h'] = df['temp'].shift(-i)
#         future_data[f'wind_f_{i}h'] = df['wind_speed'].shift(-i)

#     # --- Create Multi-Output Targets ---
#     for i in range(1, 73):
#         future_data[f'target_aqi_{i}h'] = df['aqi'].shift(-i)

#     # 4. Join all new columns at once (De-fragmented way)
#     future_df = pd.DataFrame(future_data)
#     df = pd.concat([df, future_df], axis=1)

#     # 5. CLEANUP
#     # Drop rows at the beginning (no lags) and end (no 72h future leads)
#     df = df.dropna().reset_index(drop=True)
    
#     # Drop intermediate column used for calculation
#     df = df.drop(columns=['aqi_lag_2h'])

#     return df

def compute_features(df):
    # Ensure chronological order
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 2. Winter Correction (Crucial for Karachi Smog)
    df['is_winter'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)

    # 3. Rolling Average & Lags
    df['aqi_rolling_6h'] = df['aqi'].rolling(window=6).mean()
    df['aqi_lag_1h'] = df['aqi'].shift(1)
    df['aqi_lag_2h'] = df['aqi'].shift(2)

    # 4. Safety: Change Rate (Avoiding Division by Zero)
    # Using 0.1 instead of 1e-6 for better numerical stability
    df['aqi_change_rate'] = (df['aqi_lag_1h'] - df['aqi_lag_2h']) / (df['aqi_lag_2h'] + 0.1)

    # 5. Future Leads (Forecast data)
    future_data = {}
    for i in range(1, 73):
        future_data[f'temp_f_{i}h'] = df['temp'].shift(-i)
        future_data[f'wind_f_{i}h'] = df['wind_speed'].shift(-i)
        future_data[f'target_aqi_{i}h'] = df['aqi'].shift(-i)

    future_df = pd.DataFrame(future_data)
    df = pd.concat([df, future_df], axis=1)

    # --- THE DEEP CLEAN ---
    # Drop rows at start (lags/rolling) and end (future leads)
    df = df.dropna()

    # Replace Infinity with 0 (division by zero can still produce 'inf')
    df = df.replace([np.inf, -np.inf], 0)

    # Drop columns that are no longer features or targets
    df = df.drop(columns=['aqi_lag_2h'])

    return df.reset_index(drop=True)