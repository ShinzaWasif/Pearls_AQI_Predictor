import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="Karachi AQI Watch", page_icon="🌬️", layout="wide")

# Load environment variables
load_dotenv()

# --- AQI CATEGORY HELPER ---
def get_aqi_status(aqi):
    """Returns Category Name, Color Code, and Health Advice based on US EPA standards."""
    aqi = aqi if aqi is not None else 0
    if aqi <= 50:
        return "Good", "#00E400", "Safe to be outside! Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "Air quality is acceptable. Sensitive groups should limit exertion."
    elif aqi <= 150:
        return "Unhealthy (SG)", "#FF7E00", "Sensitive groups (children/elderly) should stay indoors."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Everyone should avoid heavy outdoor activity."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience health effects."
    else:
        return "Hazardous", "#7E0023", "Emergency conditions: stay indoors and keep windows closed."

# --- DATA FETCHING ---
@st.cache_data(ttl=600) # Caches results for 10 minutes to save DB costs
def load_latest_data():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    col_name = os.getenv("MONGO_COLLECTION_NAME")

    if not all([mongo_uri, db_name, col_name]):
        st.error("🔑 Missing environment variables! Check your .env file.")
        return None

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        collection = db[col_name]
        
        # Pull the absolute latest record from Karachi
        latest = collection.find_one(
        {"target_day_1_aqi": {"$exists": True}}, 
        sort=[("timestamp", -1)]
    )
        return latest
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

# --- APP EXECUTION ---
latest_data = load_latest_data()

# Handle empty database scenario
if latest_data is None:
    st.warning("📡 No data found in the database.")
    st.info("Run your backfill or feature pipeline to populate MongoDB.")
    st.stop() 

# --- HEADER ---
st.title("🌬️ Karachi Real-Time Air Quality Dashboard")
last_updated = latest_data.get('timestamp', 'N/A')
st.markdown(f"**Last Updated:** {last_updated}")

# --- TOP SECTION: CURRENT STATUS ---
# Checks for predicted index first, then falls back to raw aqi
raw_aqi = latest_data.get('actual_aqi_index') or latest_data.get('aqi') or 0
current_aqi = round(float(raw_aqi))
status, color, advice = get_aqi_status(current_aqi)

# Main AQI Display Card
st.markdown(f"""
    <div style="background-color:{color}; padding:25px; border-radius:15px; text-align:center; border: 2px solid #333;">
        <h1 style="color: black; margin:0; font-size: 3.5rem;">{current_aqi}</h1>
        <h2 style="color: black; margin:0; text-transform: uppercase;">{status}</h2>
    </div>
    <div style="text-align:center; padding-top:15px;">
        <p style="font-size: 1.2rem;"><b>Health Advice:</b> {advice}</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# --- MIDDLE SECTION: 3-DAY FORECAST ---
st.subheader("🗓️ 3-Day Forecast")
col1, col2, col3 = st.columns(3)

def render_forecast_col(column, label, key):
    val = latest_data.get(key)
    if val is not None:
        val = round(float(val))
        f_status, f_color, _ = get_aqi_status(val)
        with column:
            st.metric(label=label, value=f"{val} AQI")
            st.markdown(f"<p style='color:{f_color}; font-weight:bold; font-size:1.1rem;'>● {f_status}</p>", unsafe_allow_html=True)
    else:
        column.info(f"**{label}:** No forecast data yet")

render_forecast_col(col1, "Tomorrow", "target_day_1_aqi")
render_forecast_col(col2, "Day After", "target_day_2_aqi")
render_forecast_col(col3, "In 3 Days", "target_day_3_aqi")

# --- SIDEBAR: WEATHER CONTEXT ---
with st.sidebar:
    st.header("📍 Location Context")
    st.write("City: Karachi, Pakistan")
    st.write(f"Lat/Lon: 24.86, 67.00")
    st.divider()
    st.header("🌡️ Current Weather")
    st.metric("Temperature", f"{latest_data.get('temp', 'N/A')}°C")
    st.metric("Wind Speed", f"{latest_data.get('wind_speed', 'N/A')} km/h")
    st.metric("Humidity", f"{latest_data.get('humidity', 'N/A')}%")