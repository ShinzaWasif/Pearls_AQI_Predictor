# import pandas as pd
# import numpy as np

# def calculate_usa_aqi(pm_conc):
#     """Calculates US EPA AQI for PM2.5 using 2024 breakpoints."""
#     # Truncate to 1 decimal place per EPA rules
#     cp = np.floor(pm_conc * 10) / 10
    
#     if cp <= 9.0:
#         return ((50 - 0) / (9.0 - 0.0)) * (cp - 0.0) + 0
#     elif cp <= 35.4:
#         return ((100 - 51) / (35.4 - 9.1)) * (cp - 9.1) + 51
#     elif cp <= 55.4:
#         return ((150 - 101) / (55.4 - 35.5)) * (cp - 35.5) + 101
#     elif cp <= 125.4:
#         return ((200 - 151) / (125.4 - 55.5)) * (cp - 55.5) + 151
#     elif cp <= 225.4:
#         return ((300 - 201) / (225.4 - 125.5)) * (cp - 125.5) + 201
#     elif cp <= 325.4:
#         return ((400 - 301) / (325.4 - 225.5)) * (cp - 225.5) + 301
#     else:
#         return ((500 - 401) / (500.4 - 325.5)) * (cp - 325.5) + 401

# def compute_features(df):
#     df = df.sort_values('timestamp').reset_index(drop=True)

#     # 1. Official AQI Calculation
#     # Apply the official formula to every PM2.5 concentration
#     df['aqi_calibrated'] = df['aqi'].apply(calculate_usa_aqi)

#     # 2. Time Features (Cyclic)
#     df['hour'] = df['timestamp'].dt.hour
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
#     df['month'] = df['timestamp'].dt.month
#     df['is_winter'] = df['month'].apply(lambda x: 1 if x in [11, 12, 1, 2] else 0)

#     # 3. Weather Interactions
#     # In Karachi: Humidity traps PM; Wind speed clears it.
#     df['smog_index'] = (df['humidity'] / (df['wind_speed'] + 1)) * df['is_winter']

#     # 4. Rolling Averages & Lags
#     df['aqi_rolling_6h'] = df['aqi_calibrated'].rolling(window=6).mean()
#     df['aqi_lag_1h'] = df['aqi_calibrated'].shift(1)
#     df['aqi_lag_2h'] = df['aqi_calibrated'].shift(2)
#     df['aqi_change_rate'] = (df['aqi_lag_1h'] - df['aqi_lag_2h']) / (df['aqi_lag_2h'] + 0.1)

#     # 5. Future Leads (72h Forecast)
#     future_data = {}
#     for i in range(1, 73):
#         future_data[f'temp_f_{i}h'] = df['temp'].shift(-i)
#         future_data[f'wind_f_{i}h'] = df['wind_speed'].shift(-i)
#         future_data[f'target_aqi_{i}h'] = df['aqi_calibrated'].shift(-i)

#     df = pd.concat([df, pd.DataFrame(future_data)], axis=1)

#     # Clean Up
#     df = df.dropna().replace([np.inf, -np.inf], 0)
#     df = df.drop(columns=['aqi_lag_2h', 'hour', 'month'])
#     return df.reset_index(drop=True)

import pandas as pd
import numpy as np

def calculate_usa_aqi(pm_conc):
    """Calculates US EPA AQI for PM2.5 using 2024 breakpoints."""
    cp = np.floor(pm_conc * 10) / 10
    
    if cp <= 9.0:
        return ((50 - 0) / (9.0 - 0.0)) * (cp - 0.0) + 0
    elif cp <= 35.4:
        return ((100 - 51) / (35.4 - 9.1)) * (cp - 9.1) + 51
    elif cp <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (cp - 35.5) + 101
    elif cp <= 125.4:
        return ((200 - 151) / (125.4 - 55.5)) * (cp - 55.5) + 151
    elif cp <= 225.4:
        return ((300 - 201) / (225.4 - 125.5)) * (cp - 125.5) + 201
    elif cp <= 325.4:
        return ((400 - 301) / (325.4 - 225.5)) * (cp - 225.5) + 301
    else:
        return ((500 - 401) / (500.4 - 325.5)) * (cp - 325.5) + 401

def compute_features(df, training_mode=True):
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- THE CALIBRATION STEP ---
    # Apply the 1.42 multiplier to the raw PM2.5/AQI value first
    df['aqi_calibrated_raw'] = df['aqi'] * 1.42
    
    # 1. Official AQI Calculation (using the calibrated value)
    df['aqi_calibrated'] = df['aqi_calibrated_raw'].apply(calculate_usa_aqi)
    # ----------------------------

    # 2. Time Features (Cyclic)
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['month'] = df['timestamp'].dt.month
    df['is_winter'] = df['month'].apply(lambda x: 1 if x in [11, 12, 1, 2] else 0)

    # 3. Weather Interactions
    df['smog_index'] = (df['humidity'] / (df['wind_speed'] + 1)) * df['is_winter']

    # 4. Rolling Averages & Lags (Now based on calibrated data)
    df['aqi_rolling_6h'] = df['aqi_calibrated'].rolling(window=6).mean()
    df['aqi_lag_1h'] = df['aqi_calibrated'].shift(1)
    df['aqi_lag_2h'] = df['aqi_calibrated'].shift(2)
    df['aqi_change_rate'] = (df['aqi_lag_1h'] - df['aqi_lag_2h']) / (df['aqi_lag_2h'] + 0.1)
    # 5. Future Leads (72h Forecast)
    future_data = {}
    for i in range(1, 73):
        # We always shift weather because we get these from live forecasts
        future_data[f'temp_f_{i}h'] = df['temp'].shift(-i)
        future_data[f'wind_f_{i}h'] = df['wind_speed'].shift(-i)
        
        # We ONLY shift targets if we are in training mode
        if training_mode:
            future_data[f'target_aqi_{i}h'] = df['aqi_calibrated'].shift(-i)

    df = pd.concat([df, pd.DataFrame(future_data)], axis=1)

    # --- THE FIX ---
    # In training, we drop everything that doesn't have a 72h future target.
    # In inference, we keep the latest row so we can actually predict today's future.
    if training_mode:
        df = df.dropna()
    else:
        # Just drop the first 5 rows because they don't have rolling averages yet
        df = df.iloc[5:] 

    df = df.replace([np.inf, -np.inf], 0)
    df = df.drop(columns=['aqi_lag_2h', 'hour', 'month'])
    
    return df.reset_index(drop=True)