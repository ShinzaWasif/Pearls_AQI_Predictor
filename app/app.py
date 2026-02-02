
# import streamlit as st
# import pandas as pd
# from pymongo import MongoClient
# import os
# from datetime import datetime, timezone
# from dotenv import load_dotenv

# # --- CONFIG ---
# st.set_page_config(page_title="Karachi AQI Watch", page_icon="🌬️", layout="wide")

# # Load environment variables
# load_dotenv()

# # --- AQI CATEGORY HELPER ---
# def get_aqi_status(aqi):
#     """Returns Category Name, Color Code, and Health Advice based on US EPA standards."""
#     aqi = aqi if aqi is not None else 0
#     if aqi <= 50:
#         return "Good", "#00E400", "Safe to be outside! Air quality is satisfactory."
#     elif aqi <= 100:
#         return "Moderate", "#FFFF00", "Air quality is acceptable. Sensitive groups should limit exertion."
#     elif aqi <= 150:
#         return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should stay indoors."
#     elif aqi <= 200:
#         return "Unhealthy", "#FF0000", "Everyone should avoid heavy outdoor activity."
#     elif aqi <= 300:
#         return "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience health effects."
#     else:
#         return "Hazardous", "#7E0023", "Emergency conditions: stay indoors and keep windows closed."

# # --- DATA FETCHING ---
# @st.cache_data(ttl=60) # Reduced to 60s so you see your updates faster
# def load_latest_data():
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     col_name = os.getenv("MONGO_COLLECTION_NAME")

#     if not all([mongo_uri, db_name, col_name]):
#         st.error("🔑 Missing environment variables! Check your .env file.")
#         return None

#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
#         db = client[db_name]
#         collection = db[col_name]
        
#         # Strategy: Find the newest record that HAS a forecast
#         latest = collection.find_one(
#             {"target_day_1_aqi": {"$exists": True}}, 
#             sort=[("timestamp", -1)]
#         )
        
#         # Fallback: If no forecast exists at all, just get the newest weather record
#         if not latest:
#             latest = collection.find_one(sort=[("timestamp", -1)])
            
#         return latest
#     except Exception as e:
#         st.error(f"❌ Connection Error: {e}")
#         return None

# # --- APP EXECUTION ---
# latest_data = load_latest_data()

# if latest_data is None:
#     st.warning("📡 No data found in the database.")
#     st.stop() 

# # --- HEADER ---
# st.title("🌬️ Karachi Real-Time Air Quality Dashboard")

# # Handle timestamp formatting safety
# ts = latest_data.get('timestamp', 'N/A')
# if isinstance(ts, datetime):
#     last_updated_str = ts.strftime('%Y-%m-%d %H:%M')
# else:
#     last_updated_str = str(ts)

# st.markdown(f"**Last Updated:** {last_updated_str}")

# # Check for stale data (older than 24 hours)
# try:
#     if isinstance(ts, datetime):
#         diff = datetime.now(timezone.utc).replace(tzinfo=None) - ts.replace(tzinfo=None)
#         if diff.total_seconds() > 86400:
#             st.warning("⚠️ Note: The displayed data is older than 24 hours.")
# except:
#     pass

# # --- TOP SECTION: CURRENT STATUS ---
# raw_aqi = latest_data.get('actual_aqi_index') or latest_data.get('aqi') or 0
# current_aqi = round(float(raw_aqi))
# status, color, advice = get_aqi_status(current_aqi)

# st.markdown(f"""
#     <div style="background-color:{color}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #333;">
#         <h1 style="color: black; margin:0; font-size: 3.5rem;">{current_aqi}</h1>
#         <h2 style="color: black; margin:0; text-transform: uppercase;">{status}</h2>
#     </div>
#     <div style="text-align:center; padding-top:15px;">
#         <p style="font-size: 1.2rem;"><b>Health Advice:</b> {advice}</p>
#     </div>
# """, unsafe_allow_html=True)

# st.divider()

# # --- MIDDLE SECTION: 3-DAY FORECAST ---
# st.subheader("🗓️ 3-Day Forecast")
# col1, col2, col3 = st.columns(3)

# def render_forecast_col(column, label, key):
#     val = latest_data.get(key)
#     if val is not None:
#         val = round(float(val))
#         f_status, f_color, _ = get_aqi_status(val)
#         with column:
#             st.metric(label=label, value=f"{val} AQI")
#             st.markdown(f"<p style='color:{f_color}; font-weight:bold; font-size:1.1rem;'>● {f_status}</p>", unsafe_allow_html=True)
#     else:
#         column.info(f"**{label}:** No forecast data yet")

# render_forecast_col(col1, "Tomorrow", "target_day_1_aqi")
# render_forecast_col(col2, "Day After", "target_day_2_aqi")
# render_forecast_col(col3, "In 3 Days", "target_day_3_aqi")

# # --- SIDEBAR: WEATHER CONTEXT ---
# with st.sidebar:
#     st.header("📍 Location Context")
#     st.write("City: Karachi, Pakistan")
#     st.divider()
#     st.header("🌡️ Current Weather")
#     st.metric("Temperature", f"{latest_data.get('temp', 'N/A')}°C")
#     st.metric("Wind Speed", f"{latest_data.get('wind_speed', 'N/A')} km/h")
#     st.metric("Humidity", f"{latest_data.get('humidity', 'N/A')}%")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pymongo import MongoClient
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="Karachi Smog Watch", page_icon="🌬️", layout="wide")
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
        return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks and stay indoors."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Everyone should limit outdoor time. Masks are highly recommended."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience serious effects."
    else:
        return "Hazardous", "#7E0023", "Emergency conditions: Avoid all outdoor exposure."

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def load_latest_data():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        collection = db[col_name]
        
        # Strategy: Get the most recent record containing model predictions
        latest = collection.find_one(
            {"predicted_72h": {"$exists": True}}, 
            sort=[("timestamp", -1)]
        )
        # Fallback to absolute latest if no prediction found
        if not latest:
            latest = collection.find_one(sort=[("timestamp", -1)])
        return latest
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

# --- APP EXECUTION ---
latest_data = load_latest_data()

if latest_data is None:
    st.warning("📡 No data found. Ensure your data pipeline and predict.py are running.")
    st.stop()

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

# --- MIDDLE SECTION: METRICS & SMOG INDEX ---
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("🌡️ Temp", f"{latest_data.get('temp')}°C")
with col_b:
    st.metric("💧 Humidity", f"{latest_data.get('humidity')}%")
with col_c:
    st.metric("💨 Wind", f"{latest_data.get('wind_speed')} km/h")
with col_d:
    # Highlighting your new Smog Index feature
    smog_val = round(latest_data.get('smog_index', 0), 2)
    st.metric("🌫️ Smog Index", smog_val, help="Calculated based on Humidity, Wind, and Seasonality.")

st.divider()

# --- FORECAST SECTION ---
st.subheader("🔮 72-Hour Forecast Trends")

if "predicted_72h" in latest_data:
    # Create a trend chart using the hourly predictions
    forecast_values = latest_data["predicted_72h"]
    forecast_dates = pd.date_range(start=ts, periods=72, freq='H')
    
    df_forecast = pd.DataFrame({
        "Time": forecast_dates,
        "Predicted AQI": forecast_values
    })
    
    fig = px.line(df_forecast, x="Time", y="Predicted AQI", 
                  title="Hourly AQI Projection",
                  template="plotly_white",
                  color_discrete_sequence=["#FF7E00"])
    
    # Add a horizontal line for the "Unhealthy" threshold
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Summary Metrics for the next 3 days
    f_col1, f_col2, f_col3 = st.columns(3)
    
    day1_avg = np.mean(forecast_values[0:24])
    day2_avg = np.mean(forecast_values[24:48])
    day3_avg = np.mean(forecast_values[48:72])

    def show_day_card(col, label, val):
        s, c, _ = get_aqi_status(val)
        col.metric(label, f"{round(val)} AQI")
        col.markdown(f"<span style='color:{c}; font-weight:bold;'>● {s}</span>", unsafe_allow_html=True)

    show_day_card(f_col1, "Tomorrow", day1_avg)
    show_day_card(f_col2, "Day After", day2_avg)
    show_day_card(f_col3, "In 3 Days", day3_avg)
else:
    st.info("⌛ Waiting for next forecast run to generate hourly charts...")

# --- SIDEBAR ---
with st.sidebar:
    st.header("About This Data")
    st.write("This dashboard uses an **XGBoost AI model** trained on Karachi's historical air quality and weather patterns.")
    st.write("The **Smog Index** represents the atmospheric 'trapping' potential—high values indicate stagnant air likely to hold pollutants.")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()