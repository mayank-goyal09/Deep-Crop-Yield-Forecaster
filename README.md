<div align="center">

# 🌾 Deep-Crop-Yield-Forecaster — Geospatial AI Pipeline

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Outfit&weight=700&size=32&duration=3500&pause=1000&color=2ECC71&center=true&vCenter=true&multiline=true&width=900&height=100&lines=Deep+Learning+Crop+Yield+Predictions+🌾;Sentinel-2+Satellite+Data+→+Yield+Forecast;Stacked+LSTM+Neural+Network+%7C+Delhi%2FNCR)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Earth_Engine-GEE-0d1117?style=for-the-badge&logo=googleearth&logoColor=2ecc71)
![Folium](https://img.shields.io/badge/Folium-Maps-77B829?style=for-the-badge&logo=leaflet&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Crop_Yield_Forecaster-2ecc71?style=for-the-badge&labelColor=0c1445)](#)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/Deep-Crop-Yield-Forecaster?style=for-the-badge&color=ffd700)](https://github.com/mayank-goyal09/Deep-Crop-Yield-Forecaster/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/Deep-Crop-Yield-Forecaster?style=for-the-badge&color=87ceeb)](https://github.com/mayank-goyal09/Deep-Crop-Yield-Forecaster/network)

<br/>

### 🧠 **Harnessing Stacked LSTM to Predict Crop Yields across Delhi/NCR** 

### **From Raw Sentinel-2 Satellite Imagery → Actionable Agricultural Intelligence** 🌍

</div>

---

## ⚡ **THE PIPELINE AT A GLANCE**

<table>
<tr>
<td width="50%">

### 🎯 **What This Project Does**

This end-to-end **Geospatial AI Pipeline** transforms raw, open-source **Sentinel-2 satellite imagery** into actionable agricultural intelligence through a custom-built **Stacked LSTM Deep Learning model**. 

By leveraging the **Google Earth Engine (GEE)** API for cloud-based data extraction, we calculated critical vegetation indices like **NDVI** to track crop health over time. This metric is fused with meteorological data (CHIRPS precipitation) to account for environmental stress. This complex time-series data is reshaped into 3D tensors to train a neural network capable of "remembering" the cumulative growth journey of crops (like wheat), achieving a highly stable **validation loss of 0.05**.

The final result is a production-ready **Streamlit dashboard** that provides real-time yield forecasts, interactive heatmaps, and historical trend analysis—offering a powerful, cost-free tool for farmers and policymakers to optimize resource allocation and predict food security risks.

</td>
<td width="50%">

### ✨ **Key Highlights**

| Feature | Details |
|---------|---------|
| 🛰️ **Data Source** | Sentinel-2 MSI (10m) & CHIRPS (5km) via GEE |
| 🌿 **Vegetation Index** | Normalized Difference Vegetation Index (NDVI) |
| 📅 **Input Window** | 4 weeks of historical cumulative data |
| 🗺️ **Regions Covered** | Central Delhi, Gurugram, Noida, Ghaziabad, Faridabad |
| 🧪 **Deep Learning** | Stacked LSTM Network with Monte Carlo Dropout |
| 📉 **Performance** | High stability with 0.05 Validation Loss |
| 📊 **Output** | Yield prediction (MT/Ha) with 95% Confidence Intervals |
| 🎨 **UI Design** | Premium glassmorphism aesthetic dashboard |

</td>
</tr>
</table>

---

## 🌍 **REGIONS IN FOCUS (DELHI/NCR)**

<div align="center">

| 🏛️ **Central Delhi** | 🏙️ **Gurugram** | 🌆 **Noida** | 🏘️ **Ghaziabad** | 🏗️ **Faridabad** |
|:---:|:---:|:---:|:---:|:---:|
| Urban agriculture | Peri-urban farming | Mixed use | Agricultural belt | Industrial + agriculture |

</div>

---

## 🛠️ **TECHNOLOGY STACK**

<div align="center">

![Tech Stack](https://skillicons.dev/icons?i=python,tensorflow,github,vscode)

</div>

| **Category** | **Technologies** | **Purpose** |
|:------------:|:-----------------|:------------|
| 🐍 **Core Language** | Python 3.8+ | Primary development language |
| 🧠 **Deep Learning** | TensorFlow / Keras | Stacked LSTM model architecture |
| 🛰️ **Geospatial & Remote Sensing**| Google Earth Engine (EE) API | Cloud-based satellite data extraction |
| 📊 **Data Science** | Pandas, NumPy | Data manipulation, sequence engineering |
| 🎨 **Frontend** | Streamlit | Interactive web application |
| 📈 **Visualization** | Plotly, Folium | Dynamic charts, Interactive NDVI heatmaps |
| ⛅ **Environmental Data** | Sentinel-2 (Level-2A), CHIRPS | Open-source satellite & rainfall data |

---

## 🔬 **HOW THE GEOSPATIAL LSTM WORKS**

```mermaid
graph LR
    A[🛰️ Sentinel-2 & CHIRPS] -. GEE API .-> B[📥 Data Extraction]
    B --> C[🌿 Calculate NDVI]
    C --> D[🔄 4-Week Time-Series Tensors]
    D --> E[🧠 Stacked LSTM Network]
    E --> F[🌾 Yield Prediction MT/Ha]
    F --> G[📊 Streamlit Dashboard & Heatmaps]
    
    style A fill:#2ecc71,color:#fff
    style E fill:#3498db,color:#fff
    style G fill:#00f2fe,color:#000
```

### **The Pipeline Breakdown:**

<table>
<tr>
<td>

#### 🛰️ **1. Cloud-Based Extraction**
Fetch historical and real-time imagery using the **Google Earth Engine**:
- **Sentinel-2 MSI Level-2A** at 10m resolution (cloud-masked).
- **CHIRPS Precipitation** for robust environmental stress variables.
- Outputs are combined by location boundaries (ROI).

</td>
<td>

#### 🔄 **2. Sequence Engineering**
Reshape complex time-series data into 3D Tensors:
- **Input Features**: NDVI + Rainfall
- **Time Steps**: 4-week sliding window representing cumulative crop growth.
- Normalization and preparation for Recurrent networks.

</td>
</tr>
<tr>
<td>

#### 🧠 **3. Stacked LSTM Framework**
Deep learning neural network built to handle sequences:
- Two sequential **LSTM layers** that "remember" previous growth states.
- Followed by Dense layers for regression output (Yield in MT/Ha).
- Attained highly accurate **0.05 Validation Loss**.

</td>
<td>

#### 📊 **4. Real-time Inference & UI**
A production-ready interface that includes:
- Monte Carlo Dropout (Input Perturbation) to simulate a **95% Confidence Interval**.
- **Folium Interactive Maps** overlapping NDVI heatmaps.
- **Plotly Drill-Down Trends** linking rainfall directly to yield variations.

</td>
</tr>
</table>

---
