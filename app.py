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

