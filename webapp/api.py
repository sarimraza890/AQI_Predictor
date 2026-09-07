import sys
import os
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, DEFAULT_CITY, aqi_category, HAZARD_ALERT_THRESHOLD
from feature_pipeline.features import FEATURE_COLUMNS
from feature_store.store import FeatureStore
from model_registry.registry import ModelRegistry

app = Flask(__name__)
fs = FeatureStore()
registry = ModelRegistry()


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Pearls AQI Predictor REST API",
        "supported_cities": list(CITIES.keys()),
        "default_city": DEFAULT_CITY,
        "feature_store_connected": True,
    })


@app.route("/api/v1/cities", methods=["GET"])
def list_cities():
    return jsonify({"cities": CITIES})


@app.route("/api/v1/current", methods=["GET"])
def get_current_aqi():
    city = request.args.get("city", DEFAULT_CITY)
    if city not in CITIES:
        return jsonify({"error": f"City '{city}' not supported. Choose from {list(CITIES.keys())}"}), 400

    df = fs.read(city)
    if df.empty:
        return jsonify({"error": f"No data found in feature store for {city}"}), 404

    latest = df.sort_values("ts").iloc[-1]
    aqi_val = float(latest["us_aqi"])
    label, color, advice = aqi_category(aqi_val)

    return jsonify({
        "city": city,
        "timestamp": str(latest["ts"]),
        "current_aqi": round(aqi_val, 1),
        "category": label,
        "color_hex": color,
        "health_advice": advice,
        "hazard_alert": bool(aqi_val >= HAZARD_ALERT_THRESHOLD),
        "measurements": {
            "pm2_5": float(latest.get("pm2_5", 0)),
            "pm10": float(latest.get("pm10", 0)),
            "ozone": float(latest.get("ozone", 0)),
            "nitrogen_dioxide": float(latest.get("nitrogen_dioxide", 0)),
            "sulphur_dioxide": float(latest.get("sulphur_dioxide", 0)),
            "carbon_monoxide": float(latest.get("carbon_monoxide", 0)),
            "temperature_c": float(latest.get("temperature_2m", 0)),
            "relative_humidity": float(latest.get("relative_humidity_2m", 0)),
            "wind_speed_kmh": float(latest.get("wind_speed_10m", 0)),
        }
    })


@app.route("/api/v1/predict", methods=["GET"])
def predict_forecast():
    city = request.args.get("city", DEFAULT_CITY)
    model_family = request.args.get("model", "production")

    if city not in CITIES:
        return jsonify({"error": f"City '{city}' not supported. Choose from {list(CITIES.keys())}"}), 400

    df = fs.read(city)
    if df.empty:
        return jsonify({"error": f"No data available in feature store for {city}"}), 404

    latest_features = df.sort_values("ts").iloc[[-1]][FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    forecast = []
    horizons = ["day1", "day2", "day3"]
    labels = {"day1": "Tomorrow (+24h)", "day2": "In 2 Days (+48h)", "day3": "In 3 Days (+72h)"}

    for h in horizons:
        if model_family == "production":
            model, meta = registry.load_production(city, h)
        else:
            model, meta = registry.load_model(city, h, model_family)

        if model is None or meta is None:
            continue

        pred = float(model.predict(latest_features)[0])
        pred = float(np.clip(pred, 0, 500))
        rmse = float(meta["metrics"]["rmse"])
        label, color, advice = aqi_category(pred)

        forecast.append({
            "horizon": h,
            "label": labels[h],
            "predicted_aqi": round(pred, 1),
            "uncertainty_rmse": round(rmse, 2),
            "ci_lower_bound": round(max(0.0, pred - rmse), 1),
            "ci_upper_bound": round(min(500.0, pred + rmse), 1),
            "category": label,
            "color_hex": color,
            "health_advice": advice,
            "model_name": meta["model_name"],
        })

    if not forecast:
        return jsonify({"error": f"No trained models found for {city}. Run training pipeline."}), 503

    max_forecast_aqi = max(item["predicted_aqi"] for item in forecast)
    hazard_alert = bool(max_forecast_aqi >= HAZARD_ALERT_THRESHOLD)

    return jsonify({
        "city": city,
        "model_family_used": model_family,
        "forecast": forecast,
        "max_forecast_aqi": round(max_forecast_aqi, 1),
        "hazard_alert": hazard_alert,
    })


@app.route("/api/v1/simulate", methods=["POST"])
def simulate_scenario():
    data = request.get_json() or {}
    city = data.get("city", DEFAULT_CITY)

    if city not in CITIES:
        return jsonify({"error": f"City '{city}' not supported. Choose from {list(CITIES.keys())}"}), 400

    df = fs.read(city)
    if df.empty:
        return jsonify({"error": f"No data found in feature store for {city}"}), 404

    base_features = df.sort_values("ts").iloc[[-1]][FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median()).copy()

    pm_reduction = float(data.get("pm_reduction_percent", 0.0)) / 100.0
    rain_addition = float(data.get("additional_rain_mm", 0.0))
    wind_multiplier = float(data.get("wind_speed_multiplier", 1.0))

    sim_features = base_features.copy()
    sim_features["pm2_5"] = sim_features["pm2_5"] * (1.0 - pm_reduction)
    sim_features["pm10"] = sim_features["pm10"] * (1.0 - pm_reduction)
    sim_features["precipitation"] = sim_features["precipitation"] + rain_addition
    sim_features["wind_speed_10m"] = sim_features["wind_speed_10m"] * wind_multiplier

    sim_features["pollutant_load"] = (
        sim_features["pm2_5"] + sim_features["pm10"] +
        sim_features["nitrogen_dioxide"] + sim_features["sulphur_dioxide"] +
        sim_features["ozone"] + sim_features["carbon_monoxide"] / 100.0
    )

    horizons = ["day1", "day2", "day3"]
    baseline_predictions = {}
    simulated_predictions = {}

    for h in horizons:
        model, _ = registry.load_production(city, h)
        if model is not None:
            base_pred = float(np.clip(model.predict(base_features)[0], 0, 500))
            sim_pred = float(np.clip(model.predict(sim_features)[0], 0, 500))
            baseline_predictions[h] = round(base_pred, 1)
            simulated_predictions[h] = round(sim_pred, 1)

    return jsonify({
        "city": city,
        "interventions_applied": {
            "pm_reduction_percent": pm_reduction * 100,
            "additional_rain_mm": rain_addition,
            "wind_speed_multiplier": wind_multiplier,
        },
        "baseline_aqi": baseline_predictions,
        "simulated_aqi": simulated_predictions,
        "impact_delta": {h: round(simulated_predictions[h] - baseline_predictions[h], 1) for h in baseline_predictions}
    })


@app.route("/api/v1/leaderboard", methods=["GET"])
def get_leaderboard():
    city = request.args.get("city", DEFAULT_CITY)
    if city not in CITIES:
        return jsonify({"error": f"City '{city}' not supported"}), 400

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", f"training_summary_{city.lower()}.json")
    if not os.path.exists(report_path):
        return jsonify({"error": f"No training summary found for {city}"}), 404

    import json
    with open(report_path) as f:
        summary = json.load(f)

    return jsonify({"city": city, "leaderboard": summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
