import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pymongo import MongoClient
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# --- CONFIG ---
st.set_page_config(page_title="Karachi AQI Predictor", page_icon="", layout="wide")
load_dotenv()

# --- DYNAMIC CORRECTION LOGIC ---
def get_dynamic_aqi(raw_aqi, humidity):
    """Adjusts the 1.25 base factor based on Karachi's coastal humidity."""
    base_factor = 1.25
    if humidity > 85:
        # High moisture often causes sensor over-reading; we dampen the boost
        factor = base_factor * 0.85 
    elif humidity < 40:
        # Dry air means satellite/sensor might miss dust; we boost slightly
        factor = base_factor * 1.15
    else:
        factor = base_factor
    return round(raw_aqi * factor)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #004d4d; color: #e0f2f1; }
    h1, h2, h3, p { color: #ffffff !important; }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-size: 1.8rem !important; font-weight: bold !important; }
    section[data-testid="stSidebar"] { background-color: #002b2b !important; }
    .health-card {
        padding: 20px; border-radius: 12px; border-left: 10px solid;
        background-color: rgba(255, 255, 255, 0.05); margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- EPA AQI HELPER ---
def get_aqi_status(aqi):
    aqi = aqi if aqi is not None else 0
    if aqi <= 50: return "Good", "#00E400", "Air quality is satisfactory. Breathe easy!"
    elif aqi <= 100: return "Moderate", "#FFFF00", "Sensitive groups should reduce exertion."
    elif aqi <= 150: return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks."
    elif aqi <= 200: return "Unhealthy", "#FF0000", "Everyone should limit outdoor time."
    elif aqi <= 300: return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience effects."
    else: return "Hazardous", "#7E0023", "Emergency: Avoid all outdoor exposure."

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def load_dashboard_data():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        aqi_col = db[os.getenv("MONGO_COLLECTION_NAME")]
        
        now = datetime.now()
        
        # 1. Current Stats (Latest actual observation)
        history_cursor = aqi_col.find({"timestamp": {"$lte": now}, "is_predicted": {"$ne": True}}).sort("timestamp", -1).limit(2)
        aqi_history = list(history_cursor)
        
        # 2. Latest 72h Prediction
        latest_pred_record = aqi_col.find_one({"is_predicted": True}, sort=[("timestamp", -1)])
        
        # 3. Model Performance
        perf_data = db["model_performance_history"].find_one(sort=[("timestamp", -1)])
        
        # 4. Historical Data: 7 Days starting from Yesterday
        yesterday_end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        seven_days_ago = yesterday_end - timedelta(days=7)
        
        table_cursor = aqi_col.find({
            "timestamp": {"$gte": seven_days_ago, "$lte": yesterday_end},
            "is_predicted": {"$ne": True}
        }).sort("timestamp", -1)
        
        history_df = pd.DataFrame(list(table_cursor))
        
        return aqi_history, latest_pred_record, perf_data, history_df
    except Exception as e:
        st.error(f" Connection Error: {e}")
        return None, None, None, None

aqi_list, pred_record, performance, history_df = load_dashboard_data()

if not aqi_list:
    st.warning(" No real-time data found. Ensure your pipeline is feeding actual observations.")
    st.stop()

latest_data = aqi_list[0]
prev_data = aqi_list[1] if len(aqi_list) > 1 else latest_data

# --- UI: TOP SECTION ---
ts_raw = latest_data.get('timestamp', datetime.now())
is_utc_server = (time.timezone == 0)

if is_utc_server:
    karachi_tz = pytz.timezone("Asia/Karachi")
    if ts_raw.tzinfo is None:
        ts_raw = pytz.utc.localize(ts_raw)
    ts = ts_raw.astimezone(karachi_tz)
else:
    ts = ts_raw

st.title("Karachi Real-Time AQI Predictor")

if 'aqi_calibrated' in latest_data:
    current_aqi = round(float(latest_data['aqi_calibrated']))
else:
    current_aqi = get_dynamic_aqi(latest_data.get('aqi', 0), latest_data.get('humidity', 50))

status, color, advice = get_aqi_status(current_aqi)

header_col1, header_col2 = st.columns([1, 2])
with header_col1:
    st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; height:150px; border: 2px solid #00ffcc;">
            <p style="color: #333; margin:0; font-size: 0.8rem; font-weight: bold;">CALIBRATED EPA AQI</p>
            <h1 style="color: black; margin:0; font-size: 3.2rem;">{current_aqi}</h1>
            <h3 style="color: #333; margin:0; font-size: 1rem;">{status}</h3>
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown(f"""
        <div class="health-card" style="border-left-color: {color}; height:150px;">
            <h3 style="margin-top:0; font-size: 1.2rem;"> Health Advisory</h3>
            <p style="font-size: 1rem; line-height: 1.4;">{advice}</p>
            <p style="font-size: 0.75rem; opacity: 0.7;">Sync: {ts.strftime('%I:%M %p')} | Mode: Dual-Check Calibration</p>
        </div>
    """, unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
def render_metric(col, label, key, unit, inv=False):
    curr, prev = latest_data.get(key, 0), prev_data.get(key, 0)
    col.metric(label, f"{curr:.1f}{unit}", delta=f"{curr-prev:.1f}{unit}", delta_color="inverse" if (curr-prev > 0 and inv) else "normal")

render_metric(m1, " Temp", 'temp', "°C")
render_metric(m2, " Humidity", 'humidity', "%", inv=True)
render_metric(m3, " Wind", 'wind_speed', " km/h")
render_metric(m4, " Smog Index", 'smog_index', "")

st.divider()

# --- UI: FORECAST ---
st.subheader(" 72-Hour Forecast Outlook")
if pred_record and "predicted_72h" in pred_record:
    hourly_preds = pred_record["predicted_72h"]
    f_cols = st.columns(3)
    for i, label in enumerate(["Next 24h", "24h - 48h", "48h - 72h"]):
        avg_val = np.mean(hourly_preds[i*24:(i+1)*24])
        _, c, _ = get_aqi_status(avg_val)
        f_cols[i].markdown(f"<div style='background:rgba(255,255,255,0.05);padding:10px;border-radius:10px;border-top:4px solid {c};text-align:center;'><p style='margin:0;font-size:0.8rem;'>{label}</p><h3 style='margin:0;color:{c};'>{round(avg_val)} AQI</h3></div>", unsafe_allow_html=True)

    fig_fc = px.line(x=pd.date_range(start=ts, periods=72, freq='H'), y=hourly_preds)
    fig_fc.update_traces(line_color='#00ffcc', fill='tozeroy')
    fig_fc.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_fc, use_container_width=True)

# --- UI: IMPACT & PERFORMANCE ---
st.divider()
c_imp, c_perf = st.columns([1, 1])
with c_imp:
    st.subheader(" Specific Feature Importance")
    feats = ['smog_index', 'wind_speed', 'humidity', 'temp', 'aqi_lag_24h', 'aqi_change_rate']
    imp_df = pd.DataFrame([{"Feature": f, "Value": abs(latest_data.get(f, 0))} for f in feats if f in latest_data]).sort_values("Value")
    fig_imp = px.bar(imp_df, x="Value", y="Feature", orientation='h', color="Value", color_continuous_scale='Emrld', template="plotly_dark")
    fig_imp.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

with c_perf:
    st.subheader(" Model Performance Registry")
    if performance:
        m_dict = performance.get("metrics", {})
        champ = performance.get("champion_model", "XGBoost")
        rows = [{"Model": (n if n != champ else "Champion " + n), "MAE": round(v.get('MAE',0),2), "R2": round(v.get('R2',0),2), "MedAE": round(v.get('MedAE',0),2)} for n, v in m_dict.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- SIDEBAR ---
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

with st.sidebar:
    st.header(" Controls")
    if performance:
        st.success(f" Active Model: {performance.get('champion_model', 'XGBoost')}")
    
    if st.button(" Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()
    st.divider()

    st.subheader(" AQI Health Levels")
    st.markdown("""
| Range | Status |
| :--- | :--- |
| 🟢 **0 - 50** | Good |
| 🟡 **51 - 100** | Moderate |
| 🟠 **101 - 150** | Unhealthy (SG) |
| 🔴 **151 - 200** | Unhealthy |
| 🟣 **201 - 300** | Very Unhealthy |
| 🟤 **301+** | Hazardous |
    """)
    st.caption("US EPA Standard AQI Categories")
    st.divider()

    show_table = st.checkbox(" View Past 7 Days (Excl. Predictions)")

if show_table and history_df is not None:
    st.divider()
    st.subheader(" Historical Data Log (Past 7 Days starting Yesterday)")
    cols_to_show = ['timestamp', 'aqi', 'aqi_calibrated', 'temp', 'humidity', 'wind_speed', 'smog_index']
    display_df = history_df[[c for c in cols_to_show if c in history_df.columns]].copy()
    st.dataframe(display_df, use_container_width=True)