import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🌾 Delhi/NCR Yield Forecast",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Metrics grouping */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        border: 1px solid #4CAF50;
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headings and fonts */
    h1, h2, h3 {
        color: #4CAF50 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Status indicators */
    .stAlert {
        border-radius: 12px;
        border: none;
        background: rgba(255, 255, 255, 0.03);
    }
    
    /* Interactive map container */
    iframe {
        border-radius: 15px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    model_path = r'c:\my_local_data(one drive)\Attachments\Ambition course\my_all_projects\project 54 map-crop-analyzer\crop_yield_lstm_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def load_csv_data():
    csv_path = r'c:\my_local_data(one drive)\Attachments\Ambition course\my_all_projects\project 54 map-crop-analyzer\crop_yield_data_multi_roi.csv'
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Geolocation data for Delhi/NCR regions
ROI_LOCATIONS = {
    0: {"name": "Central Delhi", "lat": 28.6139, "lon": 77.2090},
    1: {"name": "Gurugram", "lat": 28.4595, "lon": 77.0266},
    2: {"name": "Noida", "lat": 28.5355, "lon": 77.3910},
    3: {"name": "Ghaziabad", "lat": 28.6692, "lon": 77.4538},
    4: {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178}
}

# --- APP LAYOUT ---

# Header Image
banner_path = r'c:\my_local_data(one drive)\Attachments\Ambition course\my_all_projects\project 54 map-crop-analyzer\assets\agricultural_dashboard.png'
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.title("🌾 Agricultural Intelligence Hub")
st.markdown("### Predicting Crop Yield with Neural Networks & Remote Sensing")

# Sidebar
st.sidebar.title("🎛️ Parameters")
df = load_csv_data()
model = load_resources()

# Date Selection
available_dates = sorted(df['date'].unique())
selected_date = st.sidebar.select_slider(
    "📅 Observation Period",
    options=available_dates,
    value=available_dates[len(available_dates)//2],
    format_func=lambda x: x.strftime('%Y-%m-%d')
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Model Insights")
st.sidebar.info("""
- **Architecture:** LSTM (Long Short-Term Memory)
- **Input Features:** NDVI, Precipitation
- **Sequence Length:** 4-Week History
- **Regions:** 5 Monitoring Areas (NCR)
""")

# Prediction Engine
def calculate_predictions(target_date):
    results = {}
    for roi_id, roi_info in ROI_LOCATIONS.items():
        # Filtering for the current ROI and target history
        history = df[(df['roi_id'] == roi_id) & (df['date'] <= target_date)].sort_values('date').tail(4)
        
        if len(history) < 4:
            results[roi_id] = None
        else:
            inputs = history[['ndvi', 'rainfall']].values.reshape(1, 4, 2)
            pred = model.predict(inputs, verbose=0)[0][0]
            results[roi_id] = float(pred)
    return results

predictions = calculate_predictions(selected_date)

# Display Key Metrics
st.markdown("#### ⚡ Real-time Forecast Summary")
cols = st.columns(len(ROI_LOCATIONS))
for idx, roi_name in enumerate(ROI_LOCATIONS.values()):
    roi_id = idx
    pred_val = predictions[roi_id]
    ndvi_val = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]['ndvi'].iloc[0]
    
    with cols[idx]:
        st.metric(
            label=roi_name['name'],
            value=f"{pred_val:.2f} MT" if pred_val else "Collecting...",
            delta=f"{ndvi_val:+.3f} NDVI",
            delta_color="normal"
        )

# Main Dashboard Content
col_map, col_chart = st.columns([2, 1])

with col_map:
    st.markdown("#### 🗺️ Interactive NDVI & Yield Map")
    m = folium.Map(location=[28.53, 77.25], zoom_start=10, tiles='CartoDB dark_matter', control_scale=True)
    
    heat_data = []
    
    for roi_id, roi_info in ROI_LOCATIONS.items():
        ndvi_curr = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]['ndvi'].iloc[0]
        heat_data.append([roi_info['lat'], roi_info['lon'], max(0, ndvi_curr)])
        
        # Popups for Recruiter Engagement
        popup_html = f"""
        <div style='font-family: sans-serif; color:#161b22;'>
            <h4 style='margin-bottom:5px;'>{roi_info['name']}</h4>
            <hr style='margin-top:0;'>
            <b>NDVI Index:</b> {ndvi_curr:.4f}<br>
            <b>Yield Forecast:</b> {predictions[roi_id]:.2f} MT/Ha<br>
            <i style='font-size: 0.8em; color: #666;'>Latitude: {roi_info['lat']}, Longitude: {roi_info['lon']}</i>
        </div>
        """
        folium.CircleMarker(
            location=[roi_info['lat'], roi_info['lon']],
            radius=15,
            popup=folium.Popup(popup_html, max_width=300),
            color="#2ecc71",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    HeatMap(heat_data, radius=40, blur=25).add_to(m)
    st_folium(m, width="100%", height=500)

with col_chart:
    st.markdown("#### 📊 Temporal Trends")
    selected_roi_id = st.selectbox("Switch View", options=list(ROI_LOCATIONS.keys()), format_func=lambda x: ROI_LOCATIONS[x]['name'])
    
    trend_df = df[df['roi_id'] == selected_roi_id].sort_values('date')
    
    # Area Chart for NDVI
    st.area_chart(trend_df.set_index('date')[['ndvi']], height=250)
    st.caption("Weekly NDVI Progression (Sentinel-2 Derived)")
    
    # Bar Chart for Rainfall
    st.bar_chart(trend_df.set_index('date')[['rainfall']], color="#3498db", height=150)
    st.caption("Weekly Rainfall (mm)")

# Experience Section
st.divider()
st.subheader("🛠️ Technical Stack & Methods")
t1, t2, t3 = st.columns(3)
with t1:
    st.write("**Remote Sensing**")
    st.markdown("Processed via Google Earth Engine API using Sentinel-2 MSI data for cloud-free mosaics.")
with t2:
    st.write("**Prediction Model**")
    st.markdown("Multi-layered LSTM architecture trained to handle time-series lag and seasonal variations.")
with t3:
    st.write("**Tech Stack**")
    st.markdown("Python • TensorFlow • Folium • Pandas • Streamlit • GIS Visualization")

# Simple Footer
st.markdown("<br><p style='text-align: center; color: gray;'>Project Showcase Implementation | Powered by LSTM & Remote Sensing</p>", unsafe_allow_html=True)
