import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, DEFAULT_CITY, aqi_category, HAZARD_ALERT_THRESHOLD, REPORTS_DIR
from feature_pipeline.features import FEATURE_COLUMNS
from feature_store.store import FeatureStore
from model_registry.registry import ModelRegistry

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    .hero-stat {
        font-size: 52px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -1px;
    }
    .stat-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .alert-box {
        padding: 16px 22px;
        border-radius: 14px;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

fs = FeatureStore()
registry = ModelRegistry()

st.sidebar.markdown("## 🍃 **Pearls AQI**")
st.sidebar.caption("Serverless End-to-End Predictive Intelligence")
st.sidebar.markdown("---")

selected_city = st.sidebar.selectbox(
    "Select Metropolitan City",
    options=list(CITIES.keys()),
    index=list(CITIES.keys()).index(DEFAULT_CITY),
)

model_mode = st.sidebar.selectbox(
    "Forecasting Model",
    options=["Production Best (Auto)", "Ridge Regression", "Random Forest", "Gradient Boosting", "Neural Network (MLP)", "Ensemble Blend (Mean)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Vulnerability Profile")
sensitive_group = st.sidebar.toggle("Sensitive Group Advisory", value=False, help="Enable for asthma, cardiovascular condition, children, or elderly")

st.sidebar.markdown("---")
st.sidebar.caption("Data Sources: Open-Meteo Air Quality & Archive API · Pluggable with Hopsworks & Vertex AI")

df = fs.read(selected_city)
if df.empty:
    with st.spinner(f"Bootstrapping feature store and models for {selected_city}..."):
        from feature_pipeline.demo_data import populate_demo_feature_store
        from training_pipeline.train import run as train_city
        populate_demo_feature_store([selected_city], days=180)
        train_city(selected_city)
        df = fs.read(selected_city)

df = df.sort_values("ts").reset_index(drop=True)
latest = df.iloc[-1]
current_aqi = float(latest["us_aqi"])
prev_aqi = float(df.iloc[-2]["us_aqi"]) if len(df) > 1 else current_aqi
delta_aqi = current_aqi - prev_aqi
cat_label, cat_color, cat_advice = aqi_category(current_aqi)

st.markdown(f"### 📍 Air Quality Intelligence — **{selected_city}, Pakistan**")
st.caption(f"Coordinates: {CITIES[selected_city]['lat']}°N, {CITIES[selected_city]['lon']}°E · Timestamp: {latest['ts']} UTC · Historical Feature Samples: {len(df):,}")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">Current US AQI</div>
        <div class="hero-stat" style="color:{cat_color}">{current_aqi:.0f}</div>
        <span class="badge" style="background:{cat_color}; color:#000000;">{cat_label}</span>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">PM2.5 Concentration</div>
        <div class="hero-stat">{latest.get('pm2_5', 0):.1f}</div>
        <span style="color:#94a3b8; font-size:12px;">µg/m³ (fine particulate)</span>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">PM10 Concentration</div>
        <div class="hero-stat">{latest.get('pm10', 0):.1f}</div>
        <span style="color:#94a3b8; font-size:12px;">µg/m³ (coarse dust)</span>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">Temperature & Humidity</div>
        <div class="hero-stat">{latest.get('temperature_2m', 0):.1f}°C</div>
        <span style="color:#94a3b8; font-size:12px;">RH: {latest.get('relative_humidity_2m', 0):.0f}%</span>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-container">
        <div class="stat-label">Wind Speed & Direction</div>
        <div class="hero-stat">{latest.get('wind_speed_10m', 0):.1f}</div>
        <span style="color:#94a3b8; font-size:12px;">km/h ({latest.get('wind_direction_10m', 0):.0f}°)</span>
    </div>
    """, unsafe_allow_html=True)

horizons = ["day1", "day2", "day3"]
day_labels = {"day1": "Day 1 (+24h)", "day2": "Day 2 (+48h)", "day3": "Day 3 (+72h)"}
model_key_mapping = {
    "Ridge Regression": "ridge_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Neural Network (MLP)": "neural_net_mlp",
}

forecast_data = []
latest_features = df.iloc[[-1]][FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

for h in horizons:
    if model_mode == "Production Best (Auto)":
        model, meta = registry.load_production(selected_city, h)
    elif model_mode == "Ensemble Blend (Mean)":
        preds_list = []
        rmses = []
        for m_name in model_key_mapping.values():
            m, mt = registry.load_model(selected_city, h, m_name)
            if m is not None:
                preds_list.append(float(m.predict(latest_features)[0]))
                rmses.append(float(mt["metrics"]["rmse"]))
        if preds_list:
            pred_val = float(np.mean(preds_list))
            rmse_val = float(np.mean(rmses))
            meta = {"model_name": "Ensemble (4-model blend)", "metrics": {"rmse": rmse_val, "mae": 0, "r2": 0}}
            model = True
        else:
            model, meta = None, None
    else:
        target_name = model_key_mapping.get(model_mode)
        model, meta = registry.load_model(selected_city, h, target_name)

    if model is not None and meta is not None:
        if model_mode != "Ensemble Blend (Mean)":
            pred_val = float(model.predict(latest_features)[0])
            rmse_val = float(meta["metrics"]["rmse"])

        pred_val = float(np.clip(pred_val, 0, 500))
        h_label, h_color, h_advice = aqi_category(pred_val)
        forecast_data.append({
            "horizon": h,
            "day": day_labels[h],
            "predicted_aqi": pred_val,
            "rmse": rmse_val,
            "lower": max(0.0, pred_val - rmse_val),
            "upper": min(500.0, pred_val + rmse_val),
            "category": h_label,
            "color": h_color,
            "model": meta["model_name"]
        })

forecast_df = pd.DataFrame(forecast_data)

peak_forecast = forecast_df["predicted_aqi"].max() if not forecast_df.empty else current_aqi
max_risk_level = max(current_aqi, peak_forecast)
hazard_triggered = (max_risk_level >= HAZARD_ALERT_THRESHOLD) or (sensitive_group and max_risk_level >= 101)

if hazard_triggered:
    alert_cat, alert_color, alert_rec = aqi_category(max_risk_level)
    st.markdown(f"""
    <div class="alert-box" style="background: {alert_color};">
        <span style="font-size: 26px;">⚠️</span>
        <div>
            <div style="font-size: 16px; font-weight: 700;">HAZARDOUS AIR QUALITY WARNING — {alert_cat.upper()} LEVEL ANTICIPATED</div>
            <div style="font-size: 13px; opacity: 0.95;">Peak AQI of {max_risk_level:.0f} detected. {alert_rec} Sensitive individuals should avoid all outdoor exertion.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

gauge_col, chart_col = st.columns([1.2, 2.8])

with gauge_col:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_aqi,
        delta={'reference': prev_aqi, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#22c55e"}},
        title={'text': f"Real-Time AQI Index<br><span style='font-size:12px;color:#94a3b8;'>{selected_city}</span>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': cat_color, 'thickness': 0.28},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 228, 0, 0.25)"},
                {'range': [51, 100], 'color': "rgba(255, 255, 0, 0.25)"},
                {'range': [101, 150], 'color': "rgba(255, 126, 0, 0.25)"},
                {'range': [151, 200], 'color': "rgba(255, 0, 0, 0.25)"},
                {'range': [201, 300], 'color': "rgba(143, 63, 151, 0.25)"},
                {'range': [301, 500], 'color': "rgba(126, 0, 35, 0.25)"},
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': HAZARD_ALERT_THRESHOLD
            }
        }
    ))
    gauge_fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge_fig, use_container_width=True)

with chart_col:
    if not forecast_df.empty:
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=list(forecast_df["day"]) + list(forecast_df["day"])[::-1],
            y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.18)",
            line=dict(width=0),
            name="Uncertainty Range (± RMSE)",
            hoverinfo="skip"
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df["day"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers+text",
            text=[f"AQI {val:.0f}" for val in forecast_df["predicted_aqi"]],
            textposition="top center",
            name="Forecast AQI",
            line=dict(color="#38bdf8", width=4),
            marker=dict(size=12, color=[row["color"] for _, row in forecast_df.iterrows()], line=dict(color="#ffffff", width=2))
        ))
        fig_forecast.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.06, line_width=0)
        fig_forecast.add_hrect(y0=51, y1=100, fillcolor="#ffff00", opacity=0.06, line_width=0)
        fig_forecast.add_hrect(y0=101, y1=150, fillcolor="#ff7e00", opacity=0.06, line_width=0)
        fig_forecast.add_hrect(y0=151, y1=200, fillcolor="#ff0000", opacity=0.06, line_width=0)
        fig_forecast.add_hrect(y0=201, y1=300, fillcolor="#8f3f97", opacity=0.06, line_width=0)

        fig_forecast.update_layout(
            title=f"<b>3-Day AQI Forecast & Uncertainty Bounds</b> ({forecast_df.iloc[0]['model']})",
            yaxis_title="US AQI",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

tab_simulate, tab_regional, tab_eda, tab_leaderboard, tab_shap, tab_export = st.tabs([
    "🧪 What-If Simulator",
    "🗺️ Regional Overview",
    "📊 Historical EDA & Anomalies",
    "🏆 Model Leaderboard",
    "🔍 Explainable AI (SHAP)",
    "📑 Reports & Export"
])

with tab_simulate:
    st.markdown("#### 🧪 Interactive Anti-Smog Policy & Weather Intervention Sandbox")
    st.caption("Simulate real-world interventions (e.g. industrial shutdowns, odd-even vehicular restrictions, artificial precipitation, wind shifts) and evaluate the immediate predicted impact on 3-day AQI.")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        pm_reduction = st.slider("Particulate Emissions Reduction (%)", min_value=0, max_value=80, value=25, step=5,
                                 help="Simulates odd-even car bans or smog lockdown")
    with s_col2:
        rain_increase = st.slider("Artificial Precipitation (mm)", min_value=0.0, max_value=20.0, value=0.0, step=1.0,
                                  help="Simulates cloud seeding or rain events")
    with s_col3:
        wind_mult = st.slider("Ventilation / Wind Multiplier", min_value=0.5, max_value=2.5, value=1.0, step=0.1,
                              help="Simulates incoming clean wind or stagnation")

    sim_features = latest_features.copy()
    sim_features["pm2_5"] = sim_features["pm2_5"] * (1.0 - (pm_reduction / 100.0))
    sim_features["pm10"] = sim_features["pm10"] * (1.0 - (pm_reduction / 100.0))
    sim_features["precipitation"] = sim_features["precipitation"] + rain_increase
    sim_features["wind_speed_10m"] = sim_features["wind_speed_10m"] * wind_mult
    sim_features["pollutant_load"] = (
        sim_features["pm2_5"] + sim_features["pm10"] +
        sim_features["nitrogen_dioxide"] + sim_features["sulphur_dioxide"] +
        sim_features["ozone"] + sim_features["carbon_monoxide"] / 100.0
    )

    sim_results = []
    for h in horizons:
        model, _ = registry.load_production(selected_city, h)
        if model is not None:
            base_p = float(np.clip(model.predict(latest_features)[0], 0, 500))
            sim_p = float(np.clip(model.predict(sim_features)[0], 0, 500))
            sim_results.append({
                "Day": day_labels[h],
                "Baseline AQI": round(base_p, 1),
                "Simulated AQI": round(sim_p, 1),
                "Net Improvement": round(base_p - sim_p, 1)
            })

    if sim_results:
        sim_df = pd.DataFrame(sim_results)
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(x=sim_df["Day"], y=sim_df["Baseline AQI"], name="Baseline Status Quo", marker_color="#ef4444"))
        fig_sim.add_trace(go.Bar(x=sim_df["Day"], y=sim_df["Simulated AQI"], name="Simulated Intervention", marker_color="#22c55e"))
        fig_sim.update_layout(barmode="group", height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              title="<b>Policy Intervention Impact: Baseline vs. Simulated Forecast</b>", yaxis_title="US AQI")
        st.plotly_chart(fig_sim, use_container_width=True)

        sc1, sc2, sc3 = st.columns(3)
        for i, row in sim_df.iterrows():
            with [sc1, sc2, sc3][i]:
                st.metric(
                    label=f"{row['Day']} Simulated Outcome",
                    value=f"{row['Simulated AQI']:.0f} AQI",
                    delta=f"-{row['Net Improvement']:.1f} AQI",
                    delta_color="normal"
                )

with tab_regional:
    st.markdown("#### 🗺️ Regional Comparative Intelligence — Karachi, Lahore & Islamabad")
    st.caption("Live comparative air quality telemetry across all three monitored metropolises.")

    reg_rows = []
    for c_name in CITIES.keys():
        c_df = fs.read(c_name)
        if not c_df.empty:
            c_latest = c_df.sort_values("ts").iloc[-1]
            c_aqi = float(c_latest["us_aqi"])
            c_cat, c_col, _ = aqi_category(c_aqi)
            reg_rows.append({
                "City": c_name,
                "Province": CITIES[c_name]["province"],
                "Latitude": CITIES[c_name]["lat"],
                "Longitude": CITIES[c_name]["lon"],
                "Current AQI": round(c_aqi, 1),
                "Category": c_cat,
                "Color": c_col,
                "PM2.5 (µg/m³)": round(float(c_latest.get("pm2_5", 0)), 1),
                "PM10 (µg/m³)": round(float(c_latest.get("pm10", 0)), 1),
                "Temp (°C)": round(float(c_latest.get("temperature_2m", 0)), 1),
                "Humidity (%)": round(float(c_latest.get("relative_humidity_2m", 0)), 0),
            })
    reg_df = pd.DataFrame(reg_rows)

    rc1, rc2 = st.columns([1.5, 1])
    with rc1:
        fig_reg_bar = px.bar(
            reg_df, x="City", y="Current AQI", color="Category",
            color_discrete_map={row["Category"]: row["Color"] for _, row in reg_df.iterrows()},
            text="Current AQI", title="<b>Current Real-Time AQI Across Metropolises</b>",
            template="plotly_dark", height=320
        )
        fig_reg_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_reg_bar, use_container_width=True)

    with rc2:
        st.dataframe(reg_df[["City", "Province", "Current AQI", "Category", "PM2.5 (µg/m³)", "Temp (°C)"]], hide_index=True, use_container_width=True)

with tab_eda:
    st.markdown("#### 📊 Exploratory Data Analysis & Anomaly Spike Detection")
    lookback_days = st.slider("Historical Lookback Window (Days)", min_value=7, max_value=90, value=30, step=7)
    recent_history = df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=lookback_days)].copy()

    fig_eda = px.line(recent_history, x="ts", y="us_aqi", title=f"<b>Hourly AQI Trajectory with Anomaly Events</b> (Last {lookback_days} Days)", template="plotly_dark")
    spikes = recent_history[recent_history["is_spike"] == 1]
    if not spikes.empty:
        fig_eda.add_scatter(
            x=spikes["ts"], y=spikes["us_aqi"], mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="diamond-open", line=dict(width=2, color="#ef4444")),
            name="Detected Pollution Spike Anomaly"
        )
    fig_eda.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", yaxis_title="US AQI")
    st.plotly_chart(fig_eda, use_container_width=True)

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.markdown("##### **Diurnal Traffic & Rush-Hour Cycle**")
        diurnal = recent_history.groupby("hour")["us_aqi"].mean().reset_index()
        fig_diurnal = px.line(diurnal, x="hour", y="us_aqi", markers=True, title="Mean AQI by Hour of Day (00:00 - 23:00)", template="plotly_dark")
        fig_diurnal.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with eda_c2:
        st.markdown("##### **Mean Pollutant Fingerprint**")
        pollutant_keys = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide"]
        poll_means = recent_history[pollutant_keys].mean().reset_index()
        poll_means.columns = ["Pollutant", "Concentration"]
        fig_poll = px.bar(poll_means, x="Pollutant", y="Concentration", template="plotly_dark", title="Average Pollutant Load (µg/m³)")
        fig_poll.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_poll, use_container_width=True)

with tab_leaderboard:
    st.markdown(f"#### 🏆 Model Performance Registry — {selected_city}")
    st.caption("Rigorous offline validation on holdout test partition (85/15 time-ordered split). Model with lowest RMSE is designated Production ⭐.")

    leaderboard_records = []
    safe_name = selected_city.lower()
    for h in horizons:
        candidates = registry.list_models(selected_city, h)
        prod_m, _ = registry.load_production(selected_city, h)

        for c in candidates:
            is_prod = "⭐ Production" if registry.load_production(selected_city, h)[1]["model_name"] == c["model_name"] else ""
            leaderboard_records.append({
                "Horizon": day_labels[h],
                "Model Architecture": c["model_name"],
                "RMSE": round(c["metrics"]["rmse"], 2),
                "MAE": round(c["metrics"]["mae"], 2),
                "R² Score": round(c["metrics"]["r2"], 3),
                "Status": is_prod,
                "Saved Timestamp": c.get("saved_at", "")[:19].replace("T", " ")
            })

    if leaderboard_records:
        lb_df = pd.DataFrame(leaderboard_records)
        st.dataframe(lb_df, hide_index=True, use_container_width=True)
    else:
        st.info("No trained models found in registry. Please trigger the training pipeline.")

with tab_shap:
    st.markdown(f"#### 🔍 Explainable AI (XAI) — SHAP Feature Attribution for {selected_city}")
    st.caption("SHAP (SHapley Additive exPlanations) isolates the marginal contribution of meteorological and pollutant features to the forecast.")

    shap_cols = st.columns(3)
    for idx, h in enumerate(horizons):
        shap_file = os.path.join(REPORTS_DIR, f"shap_{safe_name}_{h}.png")
        with shap_cols[idx]:
            st.markdown(f"##### **{day_labels[h]}**")
            if os.path.exists(shap_file):
                st.image(shap_file, use_container_width=True)
            else:
                st.warning(f"SHAP artifact for {h} pending generation.")

with tab_export:
    st.markdown("#### 📑 Deliverable Export & Telemetry Reports")
    st.caption("Download structured analytical outputs for reporting, dashboards, or external pipeline consumers.")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if not forecast_df.empty:
            forecast_export = forecast_df[["day", "predicted_aqi", "rmse", "lower", "upper", "category", "model"]]
            st.dataframe(forecast_export, hide_index=True, use_container_width=True)
            st.download_button(
                label="⬇️ Download Forecast Dataset (CSV)",
                data=forecast_export.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_name}_3day_aqi_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )

    with exp_col2:
        feature_export = df.tail(24 * 30)
        st.download_button(
            label="⬇️ Download Historical 30-Day Feature Store (CSV)",
            data=feature_export.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_name}_feature_store_30d.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 12px;'>Pearls AQI Predictor &copy; 2026 · 100% Serverless MLOps Stack · Karachi · Lahore · Islamabad</div>", unsafe_allow_html=True)
