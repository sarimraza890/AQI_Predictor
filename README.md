# Pearls AQI Predictor

An enterprise-grade, serverless-style Machine Learning system forecasting Air Quality Index (AQI) 1, 2, and 3 days ahead for **Karachi, Lahore, and Islamabad**.

Built with a modular MLOps lifecycle:
**Feature Pipeline → Feature Store → Training Pipeline → Model Registry → Web Dashboard & REST API**.

---

## 1. System Architecture

```text
External APIs (Open-Meteo, OpenWeather, AQICN)
                    │
                    ▼
          feature_pipeline/run.py
       (Hourly ingestion + feature engineering)
                    │
                    ▼
         feature_store/store.py
      (Parquet / Hopsworks / Vertex AI)
                    │
                    ▼
         training_pipeline/train.py
    (Ridge, Random Forest, GBM, MLP Neural Net)
    (RMSE, MAE, R² evaluation & SHAP attribution)
                    │
                    ▼
         model_registry/registry.py
      (Joblib weights + JSON metadata)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  webapp/app.py           webapp/api.py
(Streamlit Dashboard)   (Flask REST API)
```

---

## 2. Technology Stack

- **Core & Runtime**: Python 3.11+
- **Machine Learning**: Scikit-learn (Ridge, Random Forest, Gradient Boosting, MLP Neural Network)
- **Explainable AI**: SHAP (TreeExplainer & KernelExplainer)
- **Feature Store & Registry**: Parquet storage with pluggable Hopsworks & Vertex AI connectors
- **Orchestration & CI/CD**: GitHub Actions (hourly feature updates & daily retraining) + Apache Airflow DAGs
- **User Interfaces**: Streamlit (Executive Glassmorphic UI) + Flask (Production REST API)
- **Telemetry & Visuals**: Plotly Interactive Visuals

---

## 3. Supported Metropolises

Exclusively engineered and calibrated for Pakistan's three primary urban centers:
1. **Karachi** (`24.8607° N, 67.0011° E`): Coastal atmospheric dynamics, maritime humidity, moderate AQI baseline with winter peak pollution.
2. **Lahore** (`31.5497° N, 74.3436° E`): Continental basin subject to severe seasonal smog, thermal inversions, and agricultural biomass haze.
3. **Islamabad** (`33.6844° N, 73.0479° E`): Sub-Himalayan plateau with seasonal dust episodes and diurnal valley wind dispersion.

---

## 4. Key Capabilities & Features

### Feature Pipeline (`feature_pipeline/`)
- Ingests hourly pollutant levels (PM2.5, PM10, CO, NO₂, SO₂, O₃, US AQI) and meteorological factors (temperature, relative humidity, pressure, wind speed, wind direction, precipitation).
- Engineers 33 feature signals including:
  - **Temporal**: Hour, day, month, day of week, weekend indicator, cyclical sin/cos encodings.
  - **AQI Dynamics**: 1h and 3h rates of change, 24h rolling mean, std, and max.
  - **Atmospheric Load**: PM2.5/PM10 particulate ratio, composite pollutant load index.
  - **Anomaly Detection**: Dynamic spike detector flagging hours exceeding 2 standard deviations from recent baseline.
  - **Autoregressive Lags**: 1h, 6h, 12h, and 24h lags for AQI and temperature.
  - **Target Vectors**: Next-day, 2-day, and 3-day daily peak AQI.

### Training Pipeline (`training_pipeline/`)
- Trains four distinct model architectures per horizon:
  - **Ridge Regression**: Regularized linear baseline.
  - **Random Forest**: Non-linear ensemble bagging (250 estimators).
  - **Gradient Boosting**: Sequential gradient boosted trees (250 estimators).
  - **Deep Learning MLP**: Multi-layer perceptron neural network (64x32 hidden layers, ReLU activation, early stopping).
- Evaluates out-of-sample performance using RMSE, MAE, and R² on time-ordered splits.
- Automatically promotes best-performing model (lowest RMSE) to production.
- Generates SHAP summary plots illustrating feature importance.

### Web Dashboard (`webapp/app.py`)
- Dark glassmorphic interface with real-time radial gauge meter and EPA category indicators.
- 3-Day Forecast curve with shaded uncertainty range (± RMSE).
- Automatic hazard alert banners with sensitive group advisory toggles.
- Historical EDA with anomaly spike markers, diurnal rush-hour curves, and pollutant breakdowns.
- Model leaderboard and visual SHAP attribution tabs.
- One-click CSV and report downloads.

### Bonus Features Included
1. **What-If Policy & Weather Intervention Simulator**: Interactively test vehicular restrictions (PM reduction), cloud seeding (precipitation), or wind shifts to view real-time revised 3-day AQI forecasts.
2. **Multi-Model Selector & Ensemble**: Toggle between individual model architectures or an automated 4-model ensemble blend.
3. **Regional 3-City Comparative Matrix**: Side-by-side air quality telemetry comparing Karachi, Lahore, and Islamabad simultaneously.
4. **Production Flask REST API**: Full endpoints for headless / microservice deployment (`/api/v1/predict`, `/api/v1/current`, `/api/v1/simulate`, `/api/v1/leaderboard`).
5. **Apache Airflow DAG**: Ready-to-deploy workflow definition in `airflow/dags/pearls_aqi_pipeline_dag.py`.

---

## 5. Quickstart & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Populate Feature Store & Train Models
```bash
# Populate feature store with calibrated data for Karachi, Lahore, and Islamabad
python -m feature_pipeline.demo_data

# Train, evaluate, and promote models for all three cities
python -m training_pipeline.train --all-cities
```

### 3. Launch the Web Dashboard
```bash
streamlit run webapp/app.py
```

### 4. Launch the Flask REST API
```bash
python -m webapp.api
```

---

## 6. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status and supported cities |
| `GET` | `/api/v1/cities` | List coordinates and metadata for Karachi, Lahore, and Islamabad |
| `GET` | `/api/v1/current?city=Karachi` | Real-time AQI, EPA category, and pollutant readings |
| `GET` | `/api/v1/predict?city=Karachi` | 3-day AQI point forecast with uncertainty bounds |
| `POST` | `/api/v1/simulate` | Execute what-if intervention simulation |
| `GET` | `/api/v1/leaderboard?city=Karachi` | Model performance leaderboard (RMSE, MAE, R²) |

---

## 7. Automated Orchestration (CI/CD)

- **GitHub Actions**:
  - `.github/workflows/feature_pipeline.yml`: Ingests hourly readings for Karachi, Lahore, and Islamabad.
  - `.github/workflows/training_pipeline.yml`: Daily retraining with automated CI quality gate.
- **Apache Airflow**:
  - `airflow/dags/pearls_aqi_pipeline_dag.py`: Production DAG for scheduled execution in enterprise Airflow clusters.
