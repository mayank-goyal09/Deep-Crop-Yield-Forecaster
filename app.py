import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🌾 Delhi/NCR Yield Forecast",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS (Premium Dark Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .main { background-color: #0a0e14; color: #c5c8d0; }
    .block-container { padding-top: 1.5rem; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
        border-right: 1px solid rgba(46, 204, 113, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2ecc71 !important; font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1 { color: #ffffff !important; font-family: 'Inter', sans-serif; font-weight: 800 !important;
         background: linear-gradient(135deg, #2ecc71, #27ae60, #1abc9c);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2, h3, h4 { color: #e0e0e0 !important; font-family: 'Inter', sans-serif; font-weight: 600 !important; }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.08), rgba(26, 188, 156, 0.05));
        padding: 18px 20px; border-radius: 16px;
        border: 1px solid rgba(46, 204, 113, 0.15);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #2ecc71; transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(46, 204, 113, 0.25);
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.85rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; }

    /* Card sections */
    .glass-card {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(46, 204, 113, 0.12); border-radius: 20px;
        padding: 24px; margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        backdrop-filter: blur(16px);
    }
    .glass-card:hover { border-color: rgba(46, 204, 113, 0.3); }
    
    /* Map container */
    iframe { border-radius: 16px !important; box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(46, 204, 113, 0.08) !important;
        border-radius: 12px !important; font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.15);
    }
    .streamlit-expanderHeader:hover { border-color: #2ecc71; }
    
    /* Alerts */
    .stAlert { border-radius: 12px; border: none; background: rgba(46,204,113,0.04); }
    
    /* Hero banner */
    .hero-section {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.12) 0%, rgba(26, 188, 156, 0.08) 50%, rgba(22, 160, 133, 0.05) 100%);
        border: 1px solid rgba(46, 204, 113, 0.2); border-radius: 24px;
        padding: 36px 32px; margin-bottom: 2rem; position: relative; overflow: hidden;
    }
    .hero-section::before {
        content: ''; position: absolute; top: -50%; right: -30%;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(46, 204, 113, 0.08), transparent);
    }
    .hero-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 8px;
        background: linear-gradient(135deg, #2ecc71, #1abc9c, #16a085);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { font-size: 1.1rem; color: #8b949e; font-weight: 400; }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
        backdrop-filter: blur(10px);
    }
    .conf-high { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); }
    .conf-med { background: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid rgba(241, 196, 15, 0.3); }
    .conf-low { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.3); }
    
    /* Section dividers */
    .section-divider {
        height: 2px; margin: 2rem 0;
        background: linear-gradient(90deg, transparent, rgba(46, 204, 113, 0.3), transparent);
    }
    
    /* Footer */
    .footer-text { text-align: center; color: #4a5568; font-size: 0.85rem; padding: 2rem 0 1rem; }
    .footer-text a { color: #2ecc71; text-decoration: none; }
    
    /* Plotly chart containers */
    .js-plotly-plot { border-radius: 16px !important; }
    
    /* Latex rendering */
    .katex { font-size: 1.15em !important; color: #c5c8d0 !important; }
    
    /* Tech stack pills */
    .tech-pill {
        display: inline-block; padding: 5px 14px; margin: 3px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 500;
        background: rgba(46, 204, 113, 0.08);
        border: 1px solid rgba(46, 204, 113, 0.2);
        color: #8b949e; transition: all 0.3s ease;
    }
    .tech-pill:hover { border-color: #2ecc71; color: #2ecc71; }
</style>
""", unsafe_allow_html=True)


# ===========================
#  CACHED RESOURCES
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'crop_yield_lstm_model.h5')
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_csv_data():
    csv_path = os.path.join(BASE_DIR, 'crop_yield_data_multi_roi.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Region metadata
ROI_LOCATIONS = {
    0: {"name": "Central Delhi", "lat": 28.6139, "lon": 77.2090, "color": "#2ecc71", "emoji": "🏛️"},
    1: {"name": "Gurugram",      "lat": 28.4595, "lon": 77.0266, "color": "#3498db", "emoji": "🏙️"},
    2: {"name": "Noida",         "lat": 28.5355, "lon": 77.3910, "color": "#e67e22", "emoji": "🌆"},
    3: {"name": "Ghaziabad",     "lat": 28.6692, "lon": 77.4538, "color": "#9b59b6", "emoji": "🏘️"},
    4: {"name": "Faridabad",     "lat": 28.4089, "lon": 77.3178, "color": "#e74c3c", "emoji": "🏗️"},
}

# ===========================
#  PREDICTION ENGINE (with Confidence Intervals)
# ===========================
def predict_with_confidence(model, history_df, n_simulations=30):
    """
    Run Monte Carlo Dropout (approximation) to get prediction + uncertainty.
    Since our LSTM might not have dropout at inference, we simulate noise to
    estimate confidence intervals around the prediction.
    """
    inputs = history_df[['ndvi', 'rainfall']].values.reshape(1, 4, 2)
    
    # Base prediction
    base_pred = float(model.predict(inputs, verbose=0)[0][0])
    
    # Monte Carlo simulation: add small perturbations to inputs
    predictions = []
    for _ in range(n_simulations):
        noise = np.random.normal(0, 0.02, inputs.shape)
        noisy_input = inputs + noise
        p = float(model.predict(noisy_input, verbose=0)[0][0])
        predictions.append(p)
    
    predictions.append(base_pred)
    pred_array = np.array(predictions)
    
    mean_pred = np.mean(pred_array)
    std_pred = np.std(pred_array)
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    
    # Confidence level based on std
    if std_pred < 0.15:
        conf_level = "HIGH"
    elif std_pred < 0.35:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"
    
    return {
        "prediction": round(mean_pred, 3),
        "std": round(std_pred, 3),
        "ci_lower": round(max(0, ci_lower), 3),
        "ci_upper": round(ci_upper, 3),
        "confidence": conf_level,
    }


def calculate_all_predictions(model, df, target_date):
    """Calculate predictions for all ROIs with confidence intervals."""
    results = {}
    for roi_id in ROI_LOCATIONS:
        history = df[(df['roi_id'] == roi_id) & (df['date'] <= target_date)].sort_values('date').tail(4)
        if len(history) < 4:
            results[roi_id] = None
        else:
            results[roi_id] = predict_with_confidence(model, history)
    return results


def calculate_temporal_predictions(model, df, roi_id):
    """Calculate predictions over ALL dates for a given ROI — for the interactive chart."""
    roi_df = df[df['roi_id'] == roi_id].sort_values('date')
    dates = roi_df['date'].unique()
    pred_records = []
    
    for d in dates:
        hist = roi_df[roi_df['date'] <= d].tail(4)
        if len(hist) < 4:
            continue
        inputs = hist[['ndvi', 'rainfall']].values.reshape(1, 4, 2)
        pred = float(model.predict(inputs, verbose=0)[0][0])
        
        # Quick CI with fewer simulations for speed
        preds = []
        for _ in range(10):
            noise = np.random.normal(0, 0.02, inputs.shape)
            preds.append(float(model.predict(inputs + noise, verbose=0)[0][0]))
        preds.append(pred)
        arr = np.array(preds)
        
        pred_records.append({
            'date': d,
            'yield_pred': np.mean(arr),
            'yield_upper': np.mean(arr) + 1.96 * np.std(arr),
            'yield_lower': max(0, np.mean(arr) - 1.96 * np.std(arr)),
            'ndvi': hist['ndvi'].iloc[-1],
            'rainfall': hist['rainfall'].iloc[-1],
        })
    return pd.DataFrame(pred_records)


# ===========================
#  PLOTLY CHART BUILDERS
# ===========================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#8b949e', size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode='x unified',
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    xaxis=dict(gridcolor='rgba(46,204,113,0.08)', linecolor='rgba(46,204,113,0.15)',
               zeroline=False, showgrid=True),
    yaxis=dict(gridcolor='rgba(46,204,113,0.08)', linecolor='rgba(46,204,113,0.15)',
               zeroline=False, showgrid=True),
)


def build_ndvi_yield_chart(temporal_df, region_name, region_color):
    """Dual-axis interactive Plotly chart: NDVI trend + Predicted Yield with CI band."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Confidence band for yield
    fig.add_trace(go.Scatter(
        x=temporal_df['date'], y=temporal_df['yield_upper'],
        mode='lines', line=dict(width=0), showlegend=False,
        hoverinfo='skip',
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=temporal_df['date'], y=temporal_df['yield_lower'],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(46,204,113,0.1)', name='95% CI', showlegend=True,
        hoverinfo='skip',
    ), secondary_y=True)
    
    # NDVI line
    fig.add_trace(go.Scatter(
        x=temporal_df['date'], y=temporal_df['ndvi'],
        mode='lines+markers', name='NDVI Index',
        line=dict(color='#1abc9c', width=2.5, shape='spline'),
        marker=dict(size=5, color='#1abc9c', line=dict(color='#0a0e14', width=1.5)),
        hovertemplate='<b>NDVI</b>: %{y:.4f}<extra></extra>',
    ), secondary_y=False)
    
    # Yield prediction line
    fig.add_trace(go.Scatter(
        x=temporal_df['date'], y=temporal_df['yield_pred'],
        mode='lines+markers', name='Yield Forecast (MT/Ha)',
        line=dict(color=region_color, width=3, shape='spline'),
        marker=dict(size=6, symbol='diamond', color=region_color,
                    line=dict(color='#0a0e14', width=1.5)),
        hovertemplate='<b>Yield</b>: %{y:.3f} MT/Ha<extra></extra>',
    ), secondary_y=True)
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f'Temporal Trends — {region_name}', font=dict(size=16, color='#e0e0e0')),
        height=380,
    )
    fig.update_yaxes(title_text='NDVI Index', secondary_y=False,
                     title_font=dict(color='#1abc9c'), tickfont=dict(color='#1abc9c'))
    fig.update_yaxes(title_text='Yield (MT/Ha)', secondary_y=True,
                     title_font=dict(color=region_color), tickfont=dict(color=region_color))
    return fig


def build_rainfall_chart(temporal_df, region_name):
    """Interactive bar chart for rainfall."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=temporal_df['date'], y=temporal_df['rainfall'],
        marker=dict(
            color=temporal_df['rainfall'],
            colorscale=[[0, 'rgba(52, 152, 219, 0.3)'], [1, 'rgba(52, 152, 219, 1)']],
            line=dict(width=0),
            cornerradius=6,
        ),
        hovertemplate='<b>Rainfall</b>: %{y:.2f} mm<extra></extra>',
        name='Rainfall',
    ))
    layout = {**PLOTLY_LAYOUT}
    layout['yaxis'] = dict(title='mm', gridcolor='rgba(52,152,219,0.08)',
                           linecolor='rgba(52,152,219,0.15)', zeroline=False, showgrid=True)
    fig.update_layout(
        **layout, height=200, showlegend=False,
        title=dict(text='Weekly Rainfall (mm)', font=dict(size=13, color='#8b949e')),
    )
    return fig


def build_comparison_chart(df, model, selected_date):
    """Multi-region NDVI comparison bar chart for the selected date."""
    records = []
    for roi_id, info in ROI_LOCATIONS.items():
        row = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]
        if len(row) > 0:
            records.append({
                'Region': info['name'],
                'NDVI': row['ndvi'].iloc[0],
                'color': info['color'],
            })
    comp_df = pd.DataFrame(records)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df['Region'], y=comp_df['NDVI'],
        marker=dict(color=comp_df['color'], line=dict(width=0), cornerradius=8),
        hovertemplate='<b>%{x}</b><br>NDVI: %{y:.4f}<extra></extra>',
    ))
    layout = {**PLOTLY_LAYOUT}
    layout['yaxis'] = dict(title='NDVI', gridcolor='rgba(46,204,113,0.08)',
                           linecolor='rgba(46,204,113,0.15)', zeroline=False, showgrid=True)
    fig.update_layout(
        **layout, height=280, showlegend=False,
        title=dict(text='Cross-Region NDVI Comparison', font=dict(size=14, color='#e0e0e0')),
    )
    return fig


# ===========================
#  FOLIUM MAP BUILDER
# ===========================
def build_interactive_map(df, predictions, selected_date):
    """Build a Folium map with heatmap overlay + detailed popups, driven by date selection."""
    m = folium.Map(
        location=[28.55, 77.25], zoom_start=10,
        tiles='CartoDB dark_matter', control_scale=True,
        attr='Map tiles by CartoDB | Data: Sentinel-2 / LSTM Model'
    )
    
    heat_data = []
    
    for roi_id, roi_info in ROI_LOCATIONS.items():
        row = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]
        if len(row) == 0:
            continue
        ndvi_val = row['ndvi'].iloc[0]
        heat_data.append([roi_info['lat'], roi_info['lon'], max(0, ndvi_val)])
        
        pred = predictions.get(roi_id)
        if pred:
            yield_str = f"{pred['prediction']:.3f}"
            ci_str = f"± {pred['std']:.3f}"
            conf = pred['confidence']
            conf_color = '#2ecc71' if conf == 'HIGH' else '#f1c40f' if conf == 'MEDIUM' else '#e74c3c'
        else:
            yield_str = "Collecting..."
            ci_str = ""
            conf = "N/A"
            conf_color = '#666'
        
        popup_html = f"""
        <div style='font-family: Inter, sans-serif; min-width: 220px; padding: 4px;'>
            <div style='font-size: 16px; font-weight: 700; color: {roi_info["color"]}; margin-bottom: 6px;'>
                {roi_info['emoji']} {roi_info['name']}
            </div>
            <hr style='margin: 4px 0; border-color: #eee;'>
            <table style='width: 100%; font-size: 13px; color: #333;'>
                <tr><td><b>📅 Date</b></td><td style='text-align:right'>{selected_date.strftime('%d %b %Y')}</td></tr>
                <tr><td><b>🌿 NDVI</b></td><td style='text-align:right; font-family: monospace;'>{ndvi_val:.4f}</td></tr>
                <tr><td><b>🌾 Yield</b></td><td style='text-align:right; font-family: monospace;'>{yield_str} MT/Ha</td></tr>
                <tr><td><b>📊 CI</b></td><td style='text-align:right; font-family: monospace;'>{ci_str}</td></tr>
                <tr><td><b>✅ Confidence</b></td>
                    <td style='text-align:right;'>
                        <span style='background:{conf_color}22; color:{conf_color}; padding:2px 8px;
                               border-radius:10px; font-size:11px; font-weight:600;'>{conf}</span>
                    </td>
                </tr>
            </table>
            <div style='font-size: 10px; color: #999; margin-top: 6px;'>
                {roi_info['lat']:.4f}°N, {roi_info['lon']:.4f}°E
            </div>
        </div>
        """
        
        # Determine marker color by NDVI health
        if ndvi_val > 0.4:
            marker_color = '#2ecc71'
        elif ndvi_val > 0.15:
            marker_color = '#f1c40f'
        else:
            marker_color = '#e74c3c'
        
        folium.CircleMarker(
            location=[roi_info['lat'], roi_info['lon']],
            radius=18 + ndvi_val * 20,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{roi_info['name']} | NDVI: {ndvi_val:.3f}",
            color=marker_color,
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.55,
        ).add_to(m)
    
    # NDVI Heatmap overlay
    if heat_data:
        HeatMap(
            heat_data, radius=45, blur=30, max_zoom=13,
            gradient={0.2: '#e74c3c', 0.4: '#f39c12', 0.6: '#f1c40f', 0.8: '#2ecc71', 1: '#27ae60'}
        ).add_to(m)
    
    return m


# ===========================
#  APP LAYOUT
# ===========================

# Load resources
df = load_csv_data()
model = load_model()

# ── SIDEBAR ──────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 10px 0 20px;'>
    <span style='font-size: 2.5rem;'>🛰️</span>
    <h2 style='margin: 5px 0 0; font-size: 1.3rem; color: #2ecc71 !important;'>Control Panel</h2>
    <p style='color: #586069; font-size: 0.8rem; margin: 0;'>Adjust parameters below</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# 1) Observation Period Slider
st.sidebar.markdown("##### 📅 Observation Period")
available_dates = sorted(df['date'].unique())
selected_date = st.sidebar.select_slider(
    "Select Week",
    options=available_dates,
    value=available_dates[len(available_dates)//2],
    format_func=lambda x: x.strftime('%d %b %Y'),
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# 2) Region Selector (Multi-region support)
st.sidebar.markdown("##### 🌍 Focus Region")
region_options = {v['name']: k for k, v in ROI_LOCATIONS.items()}
selected_region_name = st.sidebar.selectbox(
    "Select Region",
    options=list(region_options.keys()),
    index=0,
    label_visibility="collapsed",
)
selected_roi_id = region_options[selected_region_name]

st.sidebar.markdown("---")

# 3) Model Info Card
st.sidebar.markdown("##### 🧠 Model Configuration")
st.sidebar.markdown("""
<div style='background: rgba(46,204,113,0.06); border: 1px solid rgba(46,204,113,0.15);
     border-radius: 14px; padding: 14px; font-size: 0.82rem; color: #8b949e;'>
    <b style='color:#2ecc71'>Architecture</b>: Stacked LSTM<br>
    <b style='color:#2ecc71'>Input</b>: NDVI + Rainfall (4 weeks)<br>
    <b style='color:#2ecc71'>Regions</b>: 5 Monitoring Areas<br>
    <b style='color:#2ecc71'>Confidence</b>: Monte Carlo Dropout
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("🔬 Deep Crop Yield Forecaster v2.0")

# ── HERO SECTION ──────────────
banner_path = os.path.join(BASE_DIR, 'assets', 'agricultural_dashboard.png')
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

st.markdown("""
<div class='hero-section'>
    <div class='hero-title'>🌾 Agricultural Intelligence Hub</div>
    <div class='hero-subtitle'>
        Real-time crop yield forecasting powered by <b>Stacked LSTM</b> neural networks
        and <b>Sentinel-2</b> satellite imagery across the Delhi/NCR region.
    </div>
</div>
""", unsafe_allow_html=True)


# ── PREDICTIONS ──────────────
with st.spinner("🔄 Running LSTM inference across all regions..."):
    predictions = calculate_all_predictions(model, df, selected_date)


# ── ⚡ REAL-TIME FORECAST SUMMARY (Clickable Metrics) ──────────────
st.markdown("#### ⚡ Real-time Forecast Summary")
st.caption(f"📅 Observation: **{selected_date.strftime('%d %B %Y')}** • Click a region on the map for more details")

cols = st.columns(len(ROI_LOCATIONS))
for idx, (roi_id, roi_info) in enumerate(ROI_LOCATIONS.items()):
    pred = predictions.get(roi_id)
    row = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]
    ndvi_val = row['ndvi'].iloc[0] if len(row) > 0 else 0
    
    with cols[idx]:
        if pred:
            conf = pred['confidence']
            conf_cls = 'conf-high' if conf == 'HIGH' else 'conf-med' if conf == 'LOW' else 'conf-med'
            
            st.metric(
                label=f"{roi_info['emoji']} {roi_info['name']}",
                value=f"{pred['prediction']:.2f} MT",
                delta=f"NDVI {ndvi_val:+.3f}",
                delta_color="normal"
            )
            st.markdown(
                f"<div style='text-align:center; margin-top:-8px;'>"
                f"<span class='confidence-badge {conf_cls}'>± {pred['std']:.3f} | {conf}</span>"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.metric(label=f"{roi_info['emoji']} {roi_info['name']}", value="Collecting...", delta="—")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── 🗺️ DYNAMIC MAP + 📊 INTERACTIVE CHARTS ──────────────
col_map, col_chart = st.columns([3, 2])

with col_map:
    st.markdown("#### 🗺️ Interactive NDVI & Yield Map")
    st.caption(f"Heatmap and markers update dynamically with the **Observation Period** slider")
    
    m = build_interactive_map(df, predictions, selected_date)
    st_folium(m, width=None, height=520, returned_objects=[])

with col_chart:
    st.markdown(f"#### 📊 Drill-Down: {ROI_LOCATIONS[selected_roi_id]['emoji']} {selected_region_name}")
    
    with st.spinner(f"Computing temporal forecast for {selected_region_name}..."):
        temporal_df = calculate_temporal_predictions(model, df, selected_roi_id)
    
    if len(temporal_df) > 0:
        # NDVI + Yield dual-axis chart
        fig_main = build_ndvi_yield_chart(
            temporal_df, selected_region_name, ROI_LOCATIONS[selected_roi_id]['color']
        )
        st.plotly_chart(fig_main, use_container_width=True, config={'displayModeBar': False})
        
        # Rainfall chart
        fig_rain = build_rainfall_chart(temporal_df, selected_region_name)
        st.plotly_chart(fig_rain, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Not enough historical data to generate trends for this region/date range.")


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── 🌍 CROSS-REGION COMPARISON ──────────────
st.markdown("#### 🌍 Cross-Region NDVI Comparison")
st.caption(f"Vegetation health index across all monitoring stations for **{selected_date.strftime('%d %b %Y')}**")

comp_col1, comp_col2 = st.columns([2, 1])

with comp_col1:
    fig_comp = build_comparison_chart(df, model, selected_date)
    st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

with comp_col2:
    st.markdown("##### 📋 Station Details")
    for roi_id, info in ROI_LOCATIONS.items():
        row = df[(df['roi_id'] == roi_id) & (df['date'] == selected_date)]
        if len(row) > 0:
            ndvi = row['ndvi'].iloc[0]
            status = "🟢 Healthy" if ndvi > 0.4 else "🟡 Moderate" if ndvi > 0.15 else "🔴 Stressed"
            pred = predictions.get(roi_id)
            yield_str = f"{pred['prediction']:.2f} MT" if pred else "—"
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
                 border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;'>
                <b style='color:{info["color"]}'>{info['emoji']} {info['name']}</b><br>
                <span style='font-size: 0.82rem; color: #8b949e;'>
                    {status} • NDVI: {ndvi:.3f} • Yield: {yield_str}
                </span>
            </div>
            """, unsafe_allow_html=True)


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── 🏗️ HOW IT WORKS (Under the Hood) ──────────────
st.markdown("#### 🏗️ Under the Hood")

with st.expander("📐 The Math: NDVI Formula", expanded=False):
    st.markdown("""
    **Normalized Difference Vegetation Index (NDVI)** quantifies vegetation health using the difference 
    between near-infrared (NIR) light — which vegetation strongly reflects — and red light — which it absorbs.
    """)
    st.latex(r"NDVI = \frac{NIR - Red}{NIR + Red}")
    st.markdown("""
    | NDVI Range | Interpretation |
    |:---:|:---|
    | **< 0.1** | Barren rock, sand, or snow |
    | **0.1 – 0.2** | Sparse vegetation or bare soil |
    | **0.2 – 0.4** | Moderate vegetation (shrubs, grassland) |
    | **0.4 – 0.6** | Dense vegetation (healthy crops) |
    | **> 0.6** | Very dense vegetation (peak growth) |
    
    *Source: Sentinel-2 MSI Level-2A (10m resolution, cloud-masked composites)*
    """)

with st.expander("🧠 The Architecture: Why Stacked LSTM?", expanded=False):
    st.markdown("""
    Our model uses a **Stacked LSTM** (Long Short-Term Memory) architecture — a type of Recurrent Neural Network
    specifically designed to learn from **sequential, time-dependent data**.
    
    #### Why LSTM over simpler models?
    
    Agricultural yield depends on **temporal patterns** — a single snapshot of NDVI tells you the *current* 
    vegetation health, but not the *trajectory*. An LSTM captures:
    
    - **Short-term weather shocks** → Sudden rainfall or drought impacts (via forget gate)
    - **Long-term growth cycles** → Seasonal planting & harvesting patterns (via cell state)
    - **Non-linear interactions** → How rainfall × NDVI jointly affect yield (via hidden layers)
    
    #### Model Pipeline
    ```
    ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
    │  Sentinel-2     │     │  4-Week      │     │  Stacked     │     │  Yield     │
    │  NDVI + Rainfall│ ──▶ │  Sliding     │ ──▶ │  LSTM        │ ──▶ │  Forecast  │
    │  (Weekly)       │     │  Window      │     │  (2 Layers)  │     │  (MT/Ha)   │
    └─────────────────┘     └──────────────┘     └──────────────┘     └────────────┘
           Input              Sequencing          Deep Learning         Output
    ```
    
    #### Confidence Estimation
    We use **Monte Carlo Input Perturbation** — small Gaussian noise is added to the input features 
    across multiple forward passes. The variance in outputs approximates the model's prediction 
    uncertainty, giving us the **95% Confidence Interval** displayed throughout the dashboard.
    
    > *"All models are wrong, but some are useful."* — George Box
    """)

with st.expander("🛰️ Data Pipeline: Google Earth Engine to Dashboard", expanded=False):
    st.markdown("""
    #### Satellite Data Acquisition
    ```
    Google Earth Engine (GEE)
      │
      ├── Sentinel-2 MSI (Level 2A, 10m)
      │     ├── filterDate(start, end)
      │     ├── filterBounds(Delhi NCR ROIs)
      │     └── Cloud masking (SCL band)
      │
      ├── CHIRPS Precipitation (5km)
      │     └── Weekly aggregation
      │
      └── Exported as CSV ──▶ Pandas DataFrame ──▶ LSTM Input
    ```
    
    #### Region of Interest (ROI) Monitoring
    Five strategic monitoring stations cover the agricultural belt:
    
    | Station | Coordinates | Land Use |
    |:---|:---:|:---|
    | Central Delhi | 28.61°N, 77.21°E | Urban agriculture |
    | Gurugram | 28.46°N, 77.03°E | Peri-urban farming |
    | Noida | 28.54°N, 77.39°E | Mixed use |
    | Ghaziabad | 28.67°N, 77.45°E | Agricultural belt |
    | Faridabad | 28.41°N, 77.32°E | Industrial + agriculture |
    """)


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── 🛠️ TECHNICAL STACK ──────────────
st.markdown("#### 🛠️ Technical Stack & Methods")

tc1, tc2, tc3 = st.columns(3)

with tc1:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 1.8rem; margin-bottom: 8px;'>🛰️</div>
        <b style='color: #2ecc71; font-size: 1rem;'>Remote Sensing</b>
        <p style='color: #8b949e; font-size: 0.85rem; margin-top: 6px;'>
            Cloud-free mosaics from <b>Sentinel-2 MSI</b> processed via 
            <b>Google Earth Engine</b> API with SCL-based cloud masking.
        </p>
        <div style='margin-top: 10px;'>
            <span class='tech-pill'>GEE</span>
            <span class='tech-pill'>Sentinel-2</span>
            <span class='tech-pill'>CHIRPS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tc2:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 1.8rem; margin-bottom: 8px;'>🧠</div>
        <b style='color: #3498db; font-size: 1rem;'>Deep Learning</b>
        <p style='color: #8b949e; font-size: 0.85rem; margin-top: 6px;'>
            <b>Stacked LSTM</b> network capturing temporal vegetation dynamics 
            with Monte Carlo uncertainty estimation for reliable forecasts.
        </p>
        <div style='margin-top: 10px;'>
            <span class='tech-pill'>TensorFlow</span>
            <span class='tech-pill'>Keras</span>
            <span class='tech-pill'>LSTM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tc3:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 1.8rem; margin-bottom: 8px;'>📊</div>
        <b style='color: #e67e22; font-size: 1rem;'>Visualization</b>
        <p style='color: #8b949e; font-size: 0.85rem; margin-top: 6px;'>
            Interactive <b>Plotly</b> charts with drill-down, <b>Folium</b> heatmaps 
            with real-time NDVI overlays, and responsive <b>Streamlit</b> interface.
        </p>
        <div style='margin-top: 10px;'>
            <span class='tech-pill'>Plotly</span>
            <span class='tech-pill'>Folium</span>
            <span class='tech-pill'>Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ──────────────
st.markdown("""
<div class='section-divider'></div>
<div class='footer-text'>
    <span style='font-size: 1.2rem;'>🌾</span><br>
    <b>Deep Crop Yield Forecaster</b> — Portfolio Showcase<br>
    Built with LSTM Neural Networks • Sentinel-2 Remote Sensing • Google Earth Engine<br>
    <span style='font-size: 0.75rem; color: #30363d;'>© 2025 Agricultural Intelligence Hub</span>
</div>
""", unsafe_allow_html=True)
