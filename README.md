<div align="center">

# 🌾 Deep-Crop-Yield-Forecaster — Geospatial AI Pipeline

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Outfit&weight=700&size=32&duration=3500&pause=1000&color=2ECC71&center=true&vCenter=true&multiline=true&width=900&height=160&lines=Deep+Learning+Crop+Yield+Predictions+🌾;Sentinel-2+Data+→+Yield+Forecast;Stacked+LSTM+Framework+%7C+Delhi+NCR)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Earth_Engine-GEE-0d1117?style=for-the-badge&logo=googleearth&logoColor=2ecc71)
![Folium](https://img.shields.io/badge/Folium-Maps-77B829?style=for-the-badge&logo=leaflet&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Crop_Yield_Forecaster-2ecc71?style=for-the-badge&labelColor=0c1445)](https://deep-crop-yield-forecaster-project.streamlit.app/)
[![📓 Google Colab](https://img.shields.io/badge/📓_GOOGLE_COLAB-Training_Pipeline-f9ab00?style=for-the-badge&labelColor=3c4043)](https://colab.research.google.com/drive/1Lwbg2jTgeWxBWDbtEuIExMHm-hxu0CaO?usp=sharing)
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

## 🎨 **DASHBOARD EXPERIENCE**

<div align="center">

### ✨ **Premium UI with Glassmorphism Design & Interactive Data**

</div>

<table>
<tr>
<td width="50%">

#### 🗺️ **Dynamic Maps & Geolocation**
- **S2 NDVI Heatmap** dynamically responds to week-by-week slider shifts.
- Detailed Folium location markers reveal predictions with **Monte Carlo confidence margins** per selected district in real-time.

</td>
<td width="50%">

#### 📊 **In-depth Time-Series Analysis**
- **Dual-Axis Line Charts** tracking vegetation health (NDVI index) alongside the predicted output in metric tons per hectare.
- **Micro-Metric Drill-downs** showing precise rainfall distribution for correlations.

</td>
</tr>
</table>

### **🎯 Design Highlights:**

```
✨ Dark mode aesthetic optimized for data density
✨ Animated glassmorphism metric cards showing confidence stats
✨ Cross-Region comparison bar-plots
✨ Explainable AI features — embedded details about mathematical NDVI equations 
✨ Seamless user interaction flow prioritizing high-tech UX
```

---

## 📂 **PROJECT STRUCTURE**

```
🌾 Deep-Crop-Yield-Forecaster/
│
├── 📊 app.py                              # Main Streamlit dashboard application
├── ⚙️ check_model.py                      # Basic script for checking architecture
│
├── 🗂️ Data & Models
│   ├── crop_yield_data_multi_roi.csv      # Processed historical 4-week window data
│   ├── crop_yield_lstm_model.h5           # Our trained core DL model Checkpoint
│
├── 📦 requirements.txt                    # Python environment dependencies
└── 📖 README.md                           # You are here! 🎉
```

---

## 🚀 **QUICK START GUIDE**

<div align="center">

</div>

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/Deep-Crop-Yield-Forecaster.git
cd Deep-Crop-Yield-Forecaster
```

### **Step 2: Create Virtual Environment** 🐍

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 3: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 4: Launch the Dashboard** 🎯

```bash
streamlit run app.py
```

### **Step 5: Open in Browser** 🌐

Navigate to: **`http://localhost:8501`**

> 🎉 **That's it!** Slide the observation dates to see historical trends, and watch the Stacked LSTM extrapolate exact yield MT figures!

---

## 🎮 **HOW TO USE THE APPLICATION**

<table>
<tr>
<td>

### 🔹 **1. Set Time Context**
Use the **Observation Period** slider in the Control Panel to pick a target historical week. This feeds the previous month's data straight into the inference engine.

</td>
<td>

### 🔹 **2. Evaluate Regional Data**
View the immediate summary numbers above the map. They identify regional yields alongside estimated **Confidence Thresholds (HIGH/MED/LOW)** computed on-the-fly.

</td>
</tr>
<tr>
<td>

### 🔹 **3. Drill Down By Region**
Select a specific Delhi/NCR region parameter to uncover temporal correlations. The system will plot historical weather and NDVI variations matching the yield trajectory.

</td>
<td>

### 🔹 **4. Explore Architecture Context**
Unfurl the "🏗️ Under the Hood" dropdowns at the bottom of the dashboard to read detailed summaries on the specific math (like Sentinel-2 MSI metrics) powering the app.

</td>
</tr>
</table>

---

## 📚 **SKILLS DEMONSTRATED**

<div align="center">

### **A Portfolio-Ready Deep Learning & Geospatial Project**

</div>

| **Category** | **Skills** |
|:-------------|:-----------|
| 🧠 **Deep Learning** | Sequence learning via Stacked LSTMs, Monte Carlo Dropout |
| 🛰️ **Geospatial AI** | Google Earth Engine API, Sentinel-2 spectral composites |
| 📊 **Time-Series Analysis** | Multi-dimensional 3D tensors, Temporal Sliding windows |
| 🐍 **Python Development** | Optimized architecture pipeline execution |
| 🎨 **UI/UX Design** | Premium application building with complex Streamlit rendering |
| 📈 **Data Visualization** | Geocoding with Folium Maps & dynamic interactions using Plotly |
| 📐 **Model Evaluation** | Rigorous loss optimization down to an incredible 0.05 index |

---

## 🔮 **FUTURE ENHANCEMENTS**

- [ ] 🌍 Broaden ROI mapping to include agricultural powerhouses (Punjab/Haryana)
- [ ] 🛰️ Merge SAR (Synthetic Aperture Radar / Sentinel-1) data for cloud-penetrating capability
- [ ] 🤖 Expand deep learning architecture with Spatio-Temporal Transformers
- [ ] 🔄 Live Earth Engine API ingestion rather than static CSV backups
- [ ] 📱 Progressive Web App (PWA) development
- [ ] 🌾 Multi-crop segmentation targeting local staples beyond standard indices

---

## 🤝 **CONTRIBUTING**

Contributions are **always welcome**! 🎉

1. 🍴 Fork the Project
2. 🌱 Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your Changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the Branch (`git push origin feature/AmazingFeature`)
5. 🎁 Open a Pull Request

---

## 📝 **LICENSE**

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 **CONNECT WITH ME**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit_Site-4facfe?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mayank-portfolio-delta.vercel.app/)
[![Email](https://img.shields.io/badge/Email-itsmaygal09@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsmaygal09@gmail.com)

<br/>

**Mayank Goyal**  
📊 Data Analyst | 🧠 Deep Learning Enthusiast | 🐍 Python Developer  
💼 Data Analyst Intern @ SpacECE Foundation India

</div>

---

## ⭐ **SHOW YOUR SUPPORT**

<div align="center">

Give a ⭐️ if this project helped you understand Geospatial Deep Learning models and inspired your agricultural forecasting projects!

<br/>

### 🌾 **Built with Deep Learning & ❤️ by Mayank Goyal**

*"Predicting tomorrow's yield, one Stacked LSTM layer at a time!"* 🧠🛰️

<br/>

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:2ecc71,100:1abc9c&height=120&section=footer)

</div>
