# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from pymongo import MongoClient
# import os
# from datetime import datetime, timezone
# from dotenv import load_dotenv

# # --- CONFIG ---
# st.set_page_config(page_title="Karachi Smog Watch", page_icon="🌬️", layout="wide")
# load_dotenv()

# # --- NEW 2024 EPA AQI HELPER ---
# def get_aqi_status(aqi):
#     """Returns Category Name, Color Code, and Health Advice based on 2024 US EPA standards."""
#     aqi = aqi if aqi is not None else 0
#     if aqi <= 50:
#         return "Good", "#00E400", "Air quality is satisfactory. Breathe easy!"
#     elif aqi <= 100:
#         return "Moderate", "#FFFF00", "Sensitive individuals should reduce outdoor exertion."
#     elif aqi <= 150:
#         return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks and stay indoors."
#     elif aqi <= 200:
#         return "Unhealthy", "#FF0000", "Everyone should limit outdoor time. Masks are highly recommended."
#     elif aqi <= 300:
#         return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience serious effects."
#     else:
#         return "Hazardous", "#7E0023", "Emergency conditions: Avoid all outdoor exposure."

# # --- DATA FETCHING ---
# @st.cache_data(ttl=60)
# def load_latest_data():
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     col_name = os.getenv("MONGO_COLLECTION_NAME")

#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
#         db = client[db_name]
#         collection = db[col_name]
        
#         # Strategy: Get the most recent record containing model predictions
#         latest = collection.find_one(
#             {"predicted_72h": {"$exists": True}}, 
#             sort=[("timestamp", -1)]
#         )
#         # Fallback to absolute latest if no prediction found
#         if not latest:
#             latest = collection.find_one(sort=[("timestamp", -1)])
#         return latest
#     except Exception as e:
#         st.error(f"❌ Connection Error: {e}")
#         return None

# # --- APP EXECUTION ---
# latest_data = load_latest_data()

# if latest_data is None:
#     st.warning("📡 No data found. Ensure your data pipeline and predict.py are running.")
#     st.stop()

# # --- HEADER ---
# st.title("🌬️ Karachi Real-Time Smog Tracker")
# ts = latest_data.get('timestamp', datetime.now())
# st.markdown(f"**Data Refreshed At:** {ts.strftime('%Y-%m-%d %I:%M %p')} PKT")

# # --- TOP SECTION: CURRENT STATUS ---
# current_aqi = round(float(latest_data.get('aqi_calibrated', 0)))
# status, color, advice = get_aqi_status(current_aqi)

# st.markdown(f"""
#     <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; border: 2px solid #ddd;">
#         <p style="color: #333; margin:0; font-size: 1.2rem; font-weight: bold;">CURRENT EPA AQI</p>
#         <h1 style="color: black; margin:0; font-size: 4.5rem;">{current_aqi}</h1>
#         <h2 style="color: #333; margin:0; text-transform: uppercase; letter-spacing: 2px;">{status}</h2>
#     </div>
#     <div style="text-align:center; padding-top:15px; margin-bottom: 30px;">
#         <p style="font-size: 1.3rem;">🛡️ <b>Health Recommendation:</b> {advice}</p>
#     </div>
# """, unsafe_allow_html=True)

# # --- MIDDLE SECTION: METRICS & SMOG INDEX ---
# col_a, col_b, col_c, col_d = st.columns(4)
# with col_a:
#     st.metric("🌡️ Temp", f"{latest_data.get('temp')}°C")
# with col_b:
#     st.metric("💧 Humidity", f"{latest_data.get('humidity')}%")
# with col_c:
#     st.metric("💨 Wind", f"{latest_data.get('wind_speed')} km/h")
# with col_d:
#     # Highlighting your new Smog Index feature
#     smog_val = round(latest_data.get('smog_index', 0), 2)
#     st.metric("🌫️ Smog Index", smog_val, help="Calculated based on Humidity, Wind, and Seasonality.")

# st.divider()

# # --- FORECAST SECTION ---
# st.subheader("🔮 72-Hour Forecast Trends")

# if "predicted_72h" in latest_data:
#     # Create a trend chart using the hourly predictions
#     forecast_values = latest_data["predicted_72h"]
#     forecast_dates = pd.date_range(start=ts, periods=72, freq='H')
    
#     df_forecast = pd.DataFrame({
#         "Time": forecast_dates,
#         "Predicted AQI": forecast_values
#     })
    
#     fig = px.line(df_forecast, x="Time", y="Predicted AQI", 
#                   title="Hourly AQI Projection",
#                   template="plotly_white",
#                   color_discrete_sequence=["#FF7E00"])
    
#     # Add a horizontal line for the "Unhealthy" threshold
#     fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy Threshold")
#     st.plotly_chart(fig, use_container_width=True)

#     # Summary Metrics for the next 3 days
#     f_col1, f_col2, f_col3 = st.columns(3)
    
#     day1_avg = np.mean(forecast_values[0:24])
#     day2_avg = np.mean(forecast_values[24:48])
#     day3_avg = np.mean(forecast_values[48:72])

#     def show_day_card(col, label, val):
#         s, c, _ = get_aqi_status(val)
#         col.metric(label, f"{round(val)} AQI")
#         col.markdown(f"<span style='color:{c}; font-weight:bold;'>● {s}</span>", unsafe_allow_html=True)

#     show_day_card(f_col1, "Tomorrow", day1_avg)
#     show_day_card(f_col2, "Day After", day2_avg)
#     show_day_card(f_col3, "In 3 Days", day3_avg)
# else:
#     st.info("⌛ Waiting for next forecast run to generate hourly charts...")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("About This Data")
#     st.write("This dashboard uses an **XGBoost AI model** trained on Karachi's historical air quality and weather patterns.")
#     st.write("The **Smog Index** represents the atmospheric 'trapping' potential—high values indicate stagnant air likely to hold pollutants.")
#     if st.button("🔄 Force Refresh"):
#         st.cache_data.clear()
#         st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pymongo import MongoClient
import os
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="Karachi Smog Watch PRO", page_icon="🌬️", layout="wide")
load_dotenv()

# --- NEW 2024 EPA AQI HELPER ---
def get_aqi_status(aqi):
    """Returns Category Name, Color Code, and Health Advice based on 2024 US EPA standards."""
    aqi = aqi if aqi is not None else 0
    if aqi <= 50:
        return "Good", "#00E400", "Air quality is satisfactory. Breathe easy!"
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "Sensitive individuals should reduce outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Everyone should limit outdoor time."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience serious effects."
    else:
        return "Hazardous", "#7E0023", "Emergency conditions: Avoid all outdoor exposure."

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def load_dashboard_data():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        aqi_col = db[os.getenv("MONGO_COLLECTION_NAME")]
        
        # 1. Get CURRENT/PAST data (Filtering out the future timestamps for the header)
        now = datetime.now()
        # This ensures we only show data that has actually happened
        history_cursor = aqi_col.find({"timestamp": {"$lte": now}}).sort("timestamp", -1).limit(2)
        aqi_history = list(history_cursor)
        
        # 2. Get the latest record that HAS a prediction (for the forecast section)
        latest_pred_record = aqi_col.find_one({"is_predicted": True}, sort=[("timestamp", -1)])
        
        # 3. Get the latest Model Performance Audit
        perf_col = db["model_performance_history"]
        perf_data = perf_col.find_one(sort=[("timestamp", -1)])
        
        return aqi_history, latest_pred_record, perf_data
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None, None, None

# --- APP EXECUTION ---
aqi_list, pred_record, performance = load_dashboard_data()

if not aqi_list:
    st.warning("📡 No historical data found in MongoDB. Ensure your feature pipeline is running.")
    st.stop()

# Use the latest historical record for the TOP metrics (ensures date is Today, not future)
latest_data = aqi_list[0]
prev_data = aqi_list[1] if len(aqi_list) > 1 else latest_data

# --- HEADER ---
st.title("🌬️ Karachi Real-Time Smog Tracker")
ts = latest_data.get('timestamp', datetime.now())
st.markdown(f"**Data Refreshed At:** {ts.strftime('%Y-%m-%d %I:%M %p')} PKT")

# --- TOP SECTION: CURRENT STATUS ---
current_aqi = round(float(latest_data.get('aqi_calibrated', 0)))
status, color, advice = get_aqi_status(current_aqi)

st.markdown(f"""
    <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; border: 2px solid #ddd;">
        <p style="color: #333; margin:0; font-size: 1.2rem; font-weight: bold;">CURRENT EPA AQI</p>
        <h1 style="color: black; margin:0; font-size: 4.5rem;">{current_aqi}</h1>
        <h2 style="color: #333; margin:0; text-transform: uppercase; letter-spacing: 2px;">{status}</h2>
    </div>
    <div style="text-align:center; padding-top:15px; margin-bottom: 30px;">
        <p style="font-size: 1.3rem;">🛡️ <b>Health Recommendation:</b> {advice}</p>
    </div>
""", unsafe_allow_html=True)

# --- MIDDLE SECTION: METRICS WITH TRENDS ---
col_a, col_b, col_c, col_d = st.columns(4)

def render_metric(col, label, key, unit, inverse=False):
    curr = latest_data.get(key, 0)
    prev = prev_data.get(key, 0)
    delta = curr - prev
    col.metric(label, f"{curr:.1f}{unit}", delta=f"{delta:.1f}{unit}", delta_color="inverse" if (delta > 0 and inverse) else "normal")

render_metric(col_a, "🌡️ Temp", 'temp', "°C")
render_metric(col_b, "💧 Humidity", 'humidity', "%", inverse=True)
render_metric(col_c, "💨 Wind Speed", 'wind_speed', " km/h")
render_metric(col_d, "🌫️ Smog Index", 'smog_index', "")

st.divider()

# --- MODEL PERFORMANCE LEADERBOARD ---
st.subheader("📊 Model Evaluation Leaderboard (Live from Registry)")
if performance:
    metrics_dict = performance.get("metrics", {})
    champion = performance.get("champion_model", "XGBoost")
    
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": "🏆 " + model_name if model_name == champion else model_name,
            "MAE (Avg Error)": round(m['MAE'], 2),
            "RMSE (Outliers)": round(m['RMSE'], 2),
            "R² Score": round(m['R2'], 2),
            "MedAE (Robust)": round(m.get('MedAE', 0), 2)
        })
    
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
else:
    st.info("No training metrics found in MongoDB.")

# --- FORECAST SECTION ---
st.subheader("🔮 72-Hour Forecast (Champion Model)")

# We look into the prediction record specifically for the forecast chart
if pred_record and "predicted_72h" in pred_record:
    hourly_preds = pred_record["predicted_72h"]
    # The forecast starts from the timestamp of the record that generated the prediction
    start_ts = pred_record.get('timestamp', datetime.now())
    forecast_dates = pd.date_range(start=start_ts, periods=72, freq='H')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=hourly_preds, 
        name="Champion Forecast", 
        line=dict(color="#FF7E00", width=4),
        fill='tozeroy',
        fillcolor='rgba(255, 126, 0, 0.1)'
    ))
    
    fig.update_layout(template="plotly_white", hovermode="x unified")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # 3-Day Summary
    st.markdown("### 🗓️ Daily Outlook")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    def show_day_card(col, label, val):
        s, c, _ = get_aqi_status(val)
        col.metric(label, f"{round(val)} AQI")
        col.markdown(f"<span style='color:{c}; font-weight:bold; font-size:1.2rem;'>● {s}</span>", unsafe_allow_html=True)

    show_day_card(f_col1, "Next 24h", np.mean(hourly_preds[0:24]))
    show_day_card(f_col2, "24h - 48h", np.mean(hourly_preds[24:48]))
    show_day_card(f_col3, "48h - 72h", np.mean(hourly_preds[48:72]))
else:
    st.info("⌛ Predictions not found. Run your Predict script to generate the 72-hour forecast.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("MLOps Controls")
    if performance:
        st.write(f"**Best Model:** {performance.get('champion_model', 'N/A')}")
    
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### EPA Guide")
    st.write("🟢 0-50: Good")
    st.write("🟡 51-100: Moderate")
    st.write("🟠 101-150: Unhealthy (SG)")
    st.write("🔴 151-200: Unhealthy")
    st.write("🟣201-300: Very Unhealthy")    
    st.write("🟤301+: Hazardous")