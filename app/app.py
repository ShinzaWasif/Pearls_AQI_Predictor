# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from pymongo import MongoClient
# from datetime import datetime
# from dotenv import load_dotenv

# # --- CONFIG ---
# st.set_page_config(page_title="Karachi Smog Watch PRO", page_icon="🌬️", layout="wide")
# load_dotenv()

# # --- CUSTOM EMERALD BLUE STYLING ---
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #004d4d; /* Deep Emerald Blue */
#         color: #e0f2f1;
#     }
#     h1, h2, h3, p {
#         color: #ffffff !important;
#     }
#     /* Optimized Metric Sizes */
#     [data-testid="stMetricValue"] {
#         color: #00ffcc !important;
#         font-size: 1.8rem !important;
#         font-weight: bold !important;
#     }
#     [data-testid="stMetricLabel"] {
#         font-size: 1rem !important;
#     }
#     section[data-testid="stSidebar"] {
#         background-color: #002b2b !important;
#     }
#     .stDataFrame {
#         background-color: #003333;
#         border: 1px solid #00ffcc;
#         border-radius: 10px;
#     }
#     /* Health Card Style */
#     .health-card {
#         padding: 20px;
#         border-radius: 12px;
#         border-left: 10px solid;
#         background-color: rgba(255, 255, 255, 0.05);
#         margin-bottom: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # --- EPA AQI HELPER ---
# def get_aqi_status(aqi):
#     aqi = aqi if aqi is not None else 0
#     if aqi <= 50:
#         return "Good", "#00E400", "Air quality is satisfactory. Breathe easy!"
#     elif aqi <= 100:
#         return "Moderate", "#FFFF00", "Sensitive individuals should reduce outdoor exertion."
#     elif aqi <= 150:
#         return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks."
#     elif aqi <= 200:
#         return "Unhealthy", "#FF0000", "Everyone should limit outdoor time."
#     elif aqi <= 300:
#         return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience serious effects."
#     else:
#         return "Hazardous", "#7E0023", "Emergency conditions: Avoid all outdoor exposure."

# # --- DATA FETCHING ---
# @st.cache_data(ttl=60)
# def load_dashboard_data():
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
#         db = client[db_name]
#         aqi_col = db[os.getenv("MONGO_COLLECTION_NAME")]
        
#         now = datetime.now()
#         history_cursor = aqi_col.find({"timestamp": {"$lte": now}}).sort("timestamp", -1).limit(2)
#         aqi_history = list(history_cursor)
        
#         latest_pred_record = aqi_col.find_one({"is_predicted": True}, sort=[("timestamp", -1)])
        
#         perf_col = db["model_performance_history"]
#         perf_data = perf_col.find_one(sort=[("timestamp", -1)])
        
#         return aqi_history, latest_pred_record, perf_data
#     except Exception as e:
#         st.error(f"❌ Connection Error: {e}")
#         return None, None, None

# aqi_list, pred_record, performance = load_dashboard_data()

# if not aqi_list:
#     st.warning("📡 No data found. Ensure your feature pipeline is running.")
#     st.stop()

# latest_data = aqi_list[0]
# prev_data = aqi_list[1] if len(aqi_list) > 1 else latest_data

# # --- 1. COMPACT TOP AREA ---
# ts = latest_data.get('timestamp', datetime.now())
# st.title("🌬️ Karachi Real-Time Smog Tracker")

# current_aqi = round(float(latest_data.get('aqi_calibrated', 0)))
# status, color, advice = get_aqi_status(current_aqi)

# # Side-by-side layout for AQI Indicator and Health Card
# header_col1, header_col2 = st.columns([1, 2])

# with header_col1:
#     st.markdown(f"""
#         <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; height:150px; border: 2px solid #00ffcc;">
#             <p style="color: #333; margin:0; font-size: 0.9rem; font-weight: bold;">CURRENT EPA AQI</p>
#             <h1 style="color: black; margin:0; font-size: 3.5rem;">{current_aqi}</h1>
#             <h3 style="color: #333; margin:0; font-size: 1.1rem;">{status}</h3>
#         </div>
#     """, unsafe_allow_html=True)

# with header_col2:
#     st.markdown(f"""
#         <div class="health-card" style="border-left-color: {color}; height:150px;">
#             <h3 style="margin-top:0; font-size: 1.3rem;">🛡️ Health Advisory</h3>
#             <p style="font-size: 1.1rem; line-height: 1.4;">{advice}</p>
#             <p style="font-size: 0.8rem; opacity: 0.7;">Last updated: {ts.strftime('%Y-%m-%d %I:%M %p')} PKT</p>
#         </div>
#     """, unsafe_allow_html=True)

# # Metrics row
# st.write("") 
# m1, m2, m3, m4 = st.columns(4)
# def render_metric(col, label, key, unit, inverse=False):
#     curr = latest_data.get(key, 0)
#     prev = prev_data.get(key, 0)
#     delta = curr - prev
#     col.metric(label, f"{curr:.1f}{unit}", delta=f"{delta:.1f}{unit}", delta_color="inverse" if (delta > 0 and inverse) else "normal")

# render_metric(m1, "🌡️ Temperature", 'temp', "°C")
# render_metric(m2, "💧 Humidity", 'humidity', "%", inverse=True)
# render_metric(m3, "💨 Wind Speed", 'wind_speed', " km/h")
# render_metric(m4, "🌫️ Smog Level", 'smog_index', "")

# st.divider()

# # --- 2. 72-HOUR FORECAST (Summary Above Line Chart) ---
# st.subheader("🔮 72-Hour Forecast Outlook (Next 3 Days)")

# if pred_record and "predicted_72h" in pred_record:
#     hourly_preds = pred_record["predicted_72h"]
    
#     # Daily Granularity Summaries PLACED ABOVE
#     f1, f2, f3 = st.columns(3)
#     def show_day_summary(col, label, val):
#         s, c, _ = get_aqi_status(val)
#         with col:
#             st.markdown(f"""
#                 <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-top: 4px solid {c}; text-align: center;">
#                     <p style="margin:0; font-size: 0.9rem; opacity: 0.8;">{label}</p>
#                     <h2 style="margin:5px 0; color: {c};">{round(val)} AQI</h2>
#                     <p style="margin:0; font-weight: bold; font-size: 0.85rem;">{s}</p>
#                 </div>
#             """, unsafe_allow_html=True)

#     show_day_summary(f1, "Day 1 (Next 24h)", np.mean(hourly_preds[0:24]))
#     show_day_summary(f2, "Day 2 (24h - 48h)", np.mean(hourly_preds[24:48]))
#     show_day_summary(f3, "Day 3 (48h - 72h)", np.mean(hourly_preds[48:72]))

#     # Line Chart
#     start_ts = pred_record.get('timestamp', datetime.now())
#     forecast_dates = pd.date_range(start=start_ts, periods=72, freq='H')
    
#     fig_fc = go.Figure()
#     fig_fc.add_trace(go.Scatter(
#         x=forecast_dates, y=hourly_preds, 
#         name="Projected AQI", line=dict(color="#00ffcc", width=4),
#         fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)'
#     ))
#     fig_fc.update_layout(
#         template="plotly_dark", 
#         hovermode="x unified", 
#         paper_bgcolor='rgba(0,0,0,0)', 
#         plot_bgcolor='rgba(0,0,0,0)',
#         height=350,
#         margin=dict(l=0, r=0, t=30, b=0)
#     )
#     st.plotly_chart(fig_fc, use_container_width=True)
# else:
#     st.info("⌛ Predictions pending...")

# st.divider()

# # --- 3. SECTION BALANCING: IMPACT & PERFORMANCE SIDE-BY-SIDE ---
# col_impact, col_leaderboard = st.columns([1, 1])

# with col_impact:
#     st.subheader("🧬 Environmental Driver Analysis")
#     backend_impact_map = {
#         'smog_index': 'Winter Humidity Trap',
#         'aqi_change_rate': 'Pollution Momentum',
#         'wind_speed': 'Wind Ventilation',
#         'temp': 'Ambient Temp',
#         'humidity': 'Moisture Level',
#         'is_winter': 'Seasonal Factor'
#     }

#     contributions = []
#     for tech_key, human_name in backend_impact_map.items():
#         val = latest_data.get(tech_key, 0)
#         if val != 0:
#             contributions.append({"Factor": human_name, "Impact Level": abs(val)})

#     impact_df = pd.DataFrame(contributions).sort_values(by="Impact Level", ascending=True)

#     fig_impact = px.bar(
#         impact_df, x="Impact Level", y="Factor", orientation='h',
#         color="Impact Level", color_continuous_scale='Emrld',
#         template="plotly_dark"
#     )
#     fig_impact.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)', 
#         plot_bgcolor='rgba(0,0,0,0)',
#         height=300,
#         margin=dict(l=0, r=0, t=0, b=0)
#     )
#     st.plotly_chart(fig_impact, use_container_width=True)

# with col_leaderboard:
#     st.subheader("📊 Model Performance Registry")
#     if performance:
#         metrics_dict = performance.get("metrics", {})
#         champion = performance.get("champion_model", "XGBoost")
#         rows = []
#         for model_name, m in metrics_dict.items():
#             rows.append({
#                 "Model": "🏆 " + model_name if model_name == champion else model_name,
#                 "MAE": round(m['MAE'], 2),
#                 "R² Score": round(m['R2'], 2),
#                 "MedAE": round(m.get('MedAE', 0), 2)
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=260)
#     else:
#         st.info("No audit data found.")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("🏢 MLOps Controls")
#     if performance:
#         st.success(f"Champion: {performance.get('champion_model', 'XGBoost')}")
#     if st.button("🔄 Refresh Data"):
#         st.cache_data.clear()
#         st.rerun()

# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from pymongo import MongoClient
# from datetime import datetime
# from dotenv import load_dotenv

# # --- CONFIG ---
# st.set_page_config(page_title="Karachi Smog Watch PRO", page_icon="🌬️", layout="wide")
# load_dotenv()

# # --- CUSTOM EMERALD BLUE STYLING ---
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #004d4d; /* Deep Emerald Blue */
#         color: #e0f2f1;
#     }
#     h1, h2, h3, p {
#         color: #ffffff !important;
#     }
#     [data-testid="stMetricValue"] {
#         color: #00ffcc !important;
#         font-size: 1.8rem !important;
#         font-weight: bold !important;
#     }
#     [data-testid="stMetricLabel"] {
#         font-size: 1rem !important;
#     }
#     section[data-testid="stSidebar"] {
#         background-color: #002b2b !important;
#     }
#     .stDataFrame {
#         background-color: #003333;
#         border: 1px solid #00ffcc;
#         border-radius: 10px;
#     }
#     .health-card {
#         padding: 20px;
#         border-radius: 12px;
#         border-left: 10px solid;
#         background-color: rgba(255, 255, 255, 0.05);
#         margin-bottom: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # --- EPA AQI HELPER ---
# def get_aqi_status(aqi):
#     aqi = aqi if aqi is not None else 0
#     if aqi <= 50:
#         return "Good", "#00E400", "Air quality is satisfactory. Breathe easy!"
#     elif aqi <= 100:
#         return "Moderate", "#FFFF00", "Sensitive individuals should reduce outdoor exertion."
#     elif aqi <= 150:
#         return "Unhealthy (SG)", "#FF7E00", "Sensitive groups should wear masks."
#     elif aqi <= 200:
#         return "Unhealthy", "#FF0000", "Everyone should limit outdoor time."
#     elif aqi <= 300:
#         return "Very Unhealthy", "#8F3F97", "Health alert: Everyone may experience serious effects."
#     else:
#         return "Hazardous", "#7E0023", "Emergency conditions: Avoid all outdoor exposure."

# # --- DATA FETCHING ---
# @st.cache_data(ttl=60)
# def load_dashboard_data():
#     mongo_uri = os.getenv("MONGO_URI")
#     db_name = os.getenv("MONGO_DB_NAME")
#     try:
#         client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
#         db = client[db_name]
#         aqi_col = db[os.getenv("MONGO_COLLECTION_NAME")]
        
#         now = datetime.now()
#         # Fetching records (including corrected aqi_calibrated from our new pipeline)
#         history_cursor = aqi_col.find({"timestamp": {"$lte": now}}).sort("timestamp", -1).limit(2)
#         aqi_history = list(history_cursor)
        
#         latest_pred_record = aqi_col.find_one({"is_predicted": True}, sort=[("timestamp", -1)])
        
#         perf_col = db["model_performance_history"]
#         perf_data = perf_col.find_one(sort=[("timestamp", -1)])
        
#         return aqi_history, latest_pred_record, perf_data
#     except Exception as e:
#         st.error(f"❌ Connection Error: {e}")
#         return None, None, None

# aqi_list, pred_record, performance = load_dashboard_data()

# if not aqi_list:
#     st.warning("📡 No data found. Ensure your feature pipeline is running.")
#     st.stop()

# latest_data = aqi_list[0]
# prev_data = aqi_list[1] if len(aqi_list) > 1 else latest_data

# # --- 1. COMPACT TOP AREA ---
# ts = latest_data.get('timestamp', datetime.now())
# st.title("🌬️ Karachi Real-Time Smog Tracker")

# # Using 'aqi_calibrated' which now includes our 1.42x Karachi Correction
# current_aqi = round(float(latest_data.get('aqi_calibrated', 0)))
# status, color, advice = get_aqi_status(current_aqi)

# header_col1, header_col2 = st.columns([1, 2])

# with header_col1:
#     st.markdown(f"""
#         <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; height:150px; border: 2px solid #00ffcc;">
#             <p style="color: #333; margin:0; font-size: 0.9rem; font-weight: bold;">CURRENT EPA AQI</p>
#             <h1 style="color: black; margin:0; font-size: 3.5rem;">{current_aqi}</h1>
#             <h3 style="color: #333; margin:0; font-size: 1.1rem;">{status}</h3>
#         </div>
#     """, unsafe_allow_html=True)

# with header_col2:
#     st.markdown(f"""
#         <div class="health-card" style="border-left-color: {color}; height:150px;">
#             <h3 style="margin-top:0; font-size: 1.3rem;">🛡️ Health Advisory</h3>
#             <p style="font-size: 1.1rem; line-height: 1.4;">{advice}</p>
#             <p style="font-size: 0.8rem; opacity: 0.7;">Last updated: {ts.strftime('%Y-%m-%d %I:%M %p')} PKT</p>
#         </div>
#     """, unsafe_allow_html=True)

# m1, m2, m3, m4 = st.columns(4)
# def render_metric(col, label, key, unit, inverse=False):
#     curr = latest_data.get(key, 0)
#     prev = prev_data.get(key, 0)
#     delta = curr - prev
#     col.metric(label, f"{curr:.1f}{unit}", delta=f"{delta:.1f}{unit}", delta_color="inverse" if (delta > 0 and inverse) else "normal")

# render_metric(m1, "🌡️ Temperature", 'temp', "°C")
# render_metric(m2, "💧 Humidity", 'humidity', "%", inverse=True)
# render_metric(m3, "💨 Wind Speed", 'wind_speed', " km/h")
# render_metric(m4, "🌫️ Smog Level", 'smog_index', "")

# st.divider()

# # --- 2. 72-HOUR FORECAST ---
# st.subheader("🔮 72-Hour Forecast Outlook (Next 3 Days)")

# if pred_record and "predicted_72h" in pred_record:
#     hourly_preds = pred_record["predicted_72h"]
    
#     f1, f2, f3 = st.columns(3)
#     def show_day_summary(col, label, val):
#         s, c, _ = get_aqi_status(val)
#         with col:
#             st.markdown(f"""
#                 <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-top: 4px solid {c}; text-align: center;">
#                     <p style="margin:0; font-size: 0.9rem; opacity: 0.8;">{label}</p>
#                     <h2 style="margin:5px 0; color: {c};">{round(val)} AQI</h2>
#                     <p style="margin:0; font-weight: bold; font-size: 0.85rem;">{s}</p>
#                 </div>
#             """, unsafe_allow_html=True)

#     show_day_summary(f1, "Day 1 (Next 24h)", np.mean(hourly_preds[0:24]))
#     show_day_summary(f2, "Day 2 (24h - 48h)", np.mean(hourly_preds[24:48]))
#     show_day_summary(f3, "Day 3 (48h - 72h)", np.mean(hourly_preds[48:72]))

#     start_ts = pred_record.get('timestamp', datetime.now())
#     forecast_dates = pd.date_range(start=start_ts, periods=72, freq='H')
    
#     fig_fc = go.Figure()
#     fig_fc.add_trace(go.Scatter(
#         x=forecast_dates, y=hourly_preds, 
#         name="Projected AQI", line=dict(color="#00ffcc", width=4),
#         fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)'
#     ))
#     fig_fc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=30, b=0),
#                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
#     st.plotly_chart(fig_fc, use_container_width=True)
# else:
#     st.info("⌛ Predictions pending...")

# st.divider()

# # --- 3. IMPACT & PERFORMANCE SIDE-BY-SIDE ---
# col_impact, col_leaderboard = st.columns([1, 1])

# with col_impact:
#     st.subheader("🧬 Environmental Driver Analysis")
    
#     # Categorizing features to show balance between Weather and Memory
#     impact_categories = {
#         'Weather Drivers': ['temp', 'humidity', 'wind_speed', 'smog_index'],
#         'Seasonal Context': ['hour_sin', 'hour_cos', 'is_winter'],
#         'Pollution Memory': ['aqi_lag_1h', 'aqi_roll_24h', 'aqi_change_rate']
#     }

#     contributions = []
#     for cat, keys in impact_categories.items():
#         # Calculating absolute impact magnitude
#         total_impact = sum([abs(latest_data.get(k, 0)) for k in keys if k in latest_data])
#         contributions.append({"Category": cat, "Impact": total_impact})

#     impact_df = pd.DataFrame(contributions)

#     # Donut chart for a clearer view of what's driving the current model "mindset"
#     fig_impact = px.pie(
#         impact_df, values="Impact", names="Category",
#         hole=0.5, color_discrete_sequence=px.colors.sequential.Emrld,
#         template="plotly_dark"
#     )
#     fig_impact.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
#                              height=300, margin=dict(l=0, r=0, t=0, b=0))
#     st.plotly_chart(fig_impact, use_container_width=True)

# with col_leaderboard:
#     st.subheader("📊 Model Performance Registry")
#     if performance:
#         metrics_dict = performance.get("metrics", {})
#         champion = performance.get("champion_model", "XGBoost")
#         rows = []
#         for model_name, m in metrics_dict.items():
#             rows.append({
#                 "Model": "🏆 " + model_name if model_name == champion else model_name,
#                 "MAE": round(m.get('MAE', 0), 2),
#                 "R² Score": round(m.get('R2', 0), 2)
#             })
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=260)
#     else:
#         st.info("No audit data found.")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("🏢 MLOps Controls")
#     if performance:
#         st.success(f"Champion: {performance.get('champion_model', 'XGBoost')}")
#     st.info("Correction Factor: 1.42x Applied")
#     if st.button("🔄 Refresh Data"):
#         st.cache_data.clear()
#         st.rerun()

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- CONFIG ---
st.set_page_config(page_title="Karachi Smog Watch PRO", page_icon="🌬️", layout="wide")
load_dotenv()

# --- DYNAMIC CORRECTION LOGIC ---
def get_dynamic_aqi(raw_aqi, humidity):
    """Adjusts the 1.42 base factor based on Karachi's coastal humidity."""
    base_factor = 1.42
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
        st.error(f"❌ Connection Error: {e}")
        return None, None, None, None

aqi_list, pred_record, performance, history_df = load_dashboard_data()

if not aqi_list:
    st.warning("📡 No real-time data found. Ensure your pipeline is feeding actual observations.")
    st.stop()

latest_data = aqi_list[0]
prev_data = aqi_list[1] if len(aqi_list) > 1 else latest_data

# --- UI: TOP SECTION ---
ts = latest_data.get('timestamp', datetime.now())
st.title("🌬️ Karachi Real-Time Smog Tracker")

# FIX: Check for 'aqi_calibrated' from DB first to get the 80s range
if 'aqi_calibrated' in latest_data:
    current_aqi = round(float(latest_data['aqi_calibrated']))
else:
    # Fallback to dynamic correction if the column doesn't exist
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
            <h3 style="margin-top:0; font-size: 1.2rem;">🛡️ Health Advisory</h3>
            <p style="font-size: 1rem; line-height: 1.4;">{advice}</p>
            <p style="font-size: 0.75rem; opacity: 0.7;">Sync: {ts.strftime('%I:%M %p')} | Mode: Dual-Check Calibration</p>
        </div>
    """, unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
def render_metric(col, label, key, unit, inv=False):
    curr, prev = latest_data.get(key, 0), prev_data.get(key, 0)
    col.metric(label, f"{curr:.1f}{unit}", delta=f"{curr-prev:.1f}{unit}", delta_color="inverse" if (curr-prev > 0 and inv) else "normal")

render_metric(m1, "🌡️ Temp", 'temp', "°C")
render_metric(m2, "💧 Humidity", 'humidity', "%", inv=True)
render_metric(m3, "💨 Wind", 'wind_speed', " km/h")
render_metric(m4, "🌫️ Smog Index", 'smog_index', "")

st.divider()

# --- UI: FORECAST ---
st.subheader("🔮 72-Hour Forecast Outlook")
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
    st.subheader("🧬 Specific Feature Importance")
    feats = ['smog_index', 'wind_speed', 'humidity', 'temp', 'aqi_lag_24h', 'aqi_change_rate']
    imp_df = pd.DataFrame([{"Feature": f, "Value": abs(latest_data.get(f, 0))} for f in feats if f in latest_data]).sort_values("Value")
    fig_imp = px.bar(imp_df, x="Value", y="Feature", orientation='h', color="Value", color_continuous_scale='Emrld', template="plotly_dark")
    fig_imp.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

with c_perf:
    st.subheader("📊 Model Performance Registry")
    if performance:
        m_dict = performance.get("metrics", {})
        champ = performance.get("champion_model", "XGBoost")
        rows = [{"Model": ("🏆 " + n if n == champ else n), "MAE": round(v.get('MAE',0),2), "R²": round(v.get('R2',0),2), "MedAE": round(v.get('MedAE',0),2)} for n, v in m_dict.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    if performance:
        st.success(f"Active Model: {performance.get('champion_model', 'XGBoost')}")
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    show_table = st.checkbox("📅 View Past 7 Days (Excl. Predictions)")

if show_table and history_df is not None:
    st.divider()
    st.subheader("📋 Historical Data Log (Past 7 Days starting Yesterday)")
    cols_to_show = ['timestamp', 'aqi', 'aqi_calibrated', 'temp', 'humidity', 'wind_speed', 'smog_index']
    display_df = history_df[[c for c in cols_to_show if c in history_df.columns]].copy()
    st.dataframe(display_df, use_container_width=True)