# 🛣️ Explainable Road Accident Risk Prediction and Route Optimization

An Explainable Artificial Intelligence (XAI) based road accident risk prediction system that analyzes historical accident data, identifies accident-prone locations (blackspots), evaluates alternative routes, and provides interpretable risk insights using Machine Learning and SHAP.

## 📌 Project Overview

Road accidents are a major public safety concern. Choosing a route based only on distance or travel time may not provide sufficient information about potential accident risks.

This project proposes an explainable machine learning-based framework that evaluates road accident risk using historical accident records, geographical information, and contextual conditions.

The system combines a Random Forest classification model, SHAP-based explainability, geospatial analysis, and interactive map visualization to help users understand and compare the estimated accident risks of alternative routes.

## 🎯 Objectives

- Predict road accident severity using Machine Learning.
- Identify accident-prone locations and geographical blackspots.
- Analyze historical accidents along alternative routes.
- Compare routes using severity-weighted accident risk.
- Explain model predictions using Explainable AI techniques.
- Visualize accident locations, blackspots, and routes through interactive maps.
- Support risk-aware route selection.

## ✨ Key Features

### 1. Machine Learning-Based Risk Prediction
- Uses a Random Forest classification model.
- Analyzes historical accident characteristics.
- Estimates the likelihood of serious or fatal accidents.

### 2. Explainable AI Using SHAP
- Uses SHAP (SHapley Additive exPlanations) to interpret model predictions.
- Identifies important factors influencing predicted risk.
- Provides global feature importance and local explanations.
- Helps users understand why a particular route receives a higher or lower risk estimate.

### 3. Accident Blackspot Detection
- Identifies accident-prone geographical regions.
- Uses geographical clustering and grid-based analysis.
- Applies severity-weighted accident analysis.
- Highlights high-risk locations on interactive maps.

### 4. Route Risk Analysis
- Accepts source and destination locations.
- Retrieves alternative driving routes.
- Analyzes accident records within a geographical corridor around each route.
- Calculates severity-weighted accident risk per kilometre.
- Compares alternative routes based on estimated risk.

### 5. Interactive Map Visualization
- Displays alternative routes.
- Highlights accident locations and blackspots.
- Provides route-related information and risk insights.
- Supports geographical exploration.

### 6. Research and Model Evaluation
- Compares different hotspot detection approaches.
- Evaluates multiple machine learning algorithms.
- Includes feature importance and ablation experiments.
- Stores research experiment results for analysis.

## 🧠 Methodology

The project follows these major stages:

1. **Data Collection:** Uses historical road accident records.
2. **Data Preprocessing:** Cleans the dataset, handles missing values, and prepares relevant features.
3. **Model Training:** Trains a Random Forest classification model.
4. **Risk Estimation:** Estimates accident severity-related risk using the trained model.
5. **Blackspot Detection:** Identifies accident-prone geographical regions.
6. **Route Analysis:** Retrieves alternative routes and examines historical accidents around them.
7. **Explainability:** Applies SHAP to interpret model predictions.
8. **Visualization:** Presents results through an interactive Streamlit dashboard.

## 📊 Risk Calculation

The route risk score considers accident severity using the following weights:

| Accident Severity | Weight |
|---|---:|
| Fatal | 10 |
| Serious | 3 |
| Slight | 1 |

The severity-weighted route risk score is calculated as:

**Risk Score = (10 × Fatal + 3 × Serious + 1 × Slight) / Route Distance (km)**

The system also considers blackspot-related penalties when ranking routes.

A lower estimated score indicates a lower historical severity-weighted accident density under this project's scoring method. It does not guarantee that a route is safer in real-world conditions.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithm | Random Forest |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Geospatial Analysis | BallTree, DBSCAN |
| Visualization | Streamlit, Folium |
| Interactive Maps | Streamlit-Folium |
| Routing | OSRM |
| Geocoding | OpenStreetMap Nominatim |
| Model Persistence | Joblib |



## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Internet connection for external routing and geocoding services

### Step 1: Clone the Repository

git clone https://github.com/vasanthikondreddy/Explainable-Road-Accident-Risk-Prediction.git


### Step 2: Navigate to the Project Directory
cd Explainable-Road-Accident-Risk-Prediction

### Step 3: Create a Virtual Environment

**Windows:**
python -m venv venv
venv\Scripts\activate


### Step 4: Install Dependencies

pip install -r requirements.txt


### Step 5: Run the Application
streamlit run app.py


The application will open in your browser.

## 🌐 External Services

The application uses external services for certain functionalities:

- **OpenStreetMap Nominatim:** Location geocoding.
- **OSRM:** Alternative driving route retrieval.

These services require an internet connection and are subject to their respective usage policies and availability.

## 🔍 Explainable AI

SHAP is used to explain the predictions generated by the machine learning model.

The explanations help identify:

- Features contributing to increased predicted risk.
- Features contributing to decreased predicted risk.
- The relative influence of individual features.
- Global patterns in model behaviour.

SHAP explanations describe how the model arrives at its predictions. They do not establish the actual causes of road accidents.

## 🚀 Future Enhancements

- Incorporate additional years of accident data.
- Improve spatial and temporal risk prediction.
- Integrate real-time traffic and road condition information.
- Improve route-risk estimation and model calibration.
- Expand geographical coverage across Indian states.
- Develop a more comprehensive road safety analytics platform.

## ⚠️ Limitations

- Predictions depend on the quality and coverage of historical accident data.
- Historical accident patterns may not represent current road conditions.
- External routing and geocoding services may experience availability limitations.
- Model predictions are estimates and should not be treated as guaranteed safety assessments.
- Further independent validation is required before real-world deployment.

## 👩‍💻 Project Information

**Project Title:** Explainable Road Accident Risk Prediction and Route Optimization

**Domain:** Machine Learning | Explainable AI | Geospatial Analytics | Road Safety
live link:**https://explainable-road-accident-risk-prediction.streamlit.app/**
