import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES
from feature_pipeline.features import engineer_features
from feature_store.store import FeatureStore


def generate_hourly(city: str, days: int = 365, seed: int = 42) -> pd.DataFrame:
    city_configs = {
        "Karachi": {"base": 75, "seasonal_amp": 35, "diurnal_amp": 12, "temp_base": 28, "humidity_base": 70},
        "Lahore": {"base": 125, "seasonal_amp": 85, "diurnal_amp": 25, "temp_base": 24, "humidity_base": 55},
        "Islamabad": {"base": 55, "seasonal_amp": 30, "diurnal_amp": 14, "temp_base": 21, "humidity_base": 50},
    }
    cfg = city_configs.get(city, {"base": 70, "seasonal_amp": 40, "diurnal_amp": 15, "temp_base": 25, "humidity_base": 60})

    rng = np.random.default_rng(abs(hash(city)) % (2**32) + seed)
    n = days * 24
    ts = pd.date_range(end=pd.Timestamp.now("UTC").tz_localize(None).floor("h"), periods=n, freq="h")

    hour = ts.hour.values
    doy = ts.dayofyear.values

    seasonal = cfg["base"] + cfg["seasonal_amp"] * np.cos(2 * np.pi * (doy - 15) / 365)
    diurnal = cfg["diurnal_amp"] * (np.sin(2 * np.pi * (hour - 8) / 24) + np.sin(2 * np.pi * (hour - 19) / 24))
    weekday_effect = np.where(ts.dayofweek.values < 5, 10, -8)

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.88 * noise[i - 1] + rng.normal(0, 5)

    spikes = np.zeros(n)
    spike_count = days // 15
    for _ in range(spike_count):
        start = rng.integers(0, n - 48)
        duration = rng.integers(6, 36)
        magnitude = rng.uniform(40, 160)
        spikes[start:start + duration] += magnitude * np.exp(-np.linspace(0, 3, duration))

    us_aqi = np.clip(seasonal + diurnal + weekday_effect + noise + spikes, 15, 490)

    pm2_5 = np.clip(us_aqi * 0.58 + rng.normal(0, 3, n), 2, None)
    pm10 = np.clip(pm2_5 * 1.65 + rng.normal(0, 5, n), 4, None)
    co = np.clip(220 + us_aqi * 2.8 + rng.normal(0, 40, n), 50, None)
    no2 = np.clip(12 + us_aqi * 0.28 + rng.normal(0, 3, n), 1, None)
    so2 = np.clip(6 + us_aqi * 0.12 + rng.normal(0, 2, n), 0.5, None)
    ozone = np.clip(25 + 22 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 4, n), 1, None)

    temp = cfg["temp_base"] + 9 * np.cos(2 * np.pi * (doy - 200) / 365) + 5 * np.sin(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 1.2, n)
    humidity = np.clip(cfg["humidity_base"] - 0.5 * (temp - cfg["temp_base"]) + rng.normal(0, 6, n), 12, 98)
    pressure = 1012 + rng.normal(0, 3, n)
    wind_speed = np.clip(7 + rng.normal(0, 3.5, n) - 0.015 * (us_aqi - 60), 0.5, None)
    wind_dir = rng.uniform(0, 360, n)
    precip = np.clip(rng.exponential(0.25, n) - 0.22, 0, None)

    us_aqi = np.clip(us_aqi - precip * 18, 10, 490)

    df = pd.DataFrame({
        "ts": pd.to_datetime(ts),
        "pm10": pm10,
        "pm2_5": pm2_5,
        "carbon_monoxide": co,
        "nitrogen_dioxide": no2,
        "sulphur_dioxide": so2,
        "ozone": ozone,
        "us_aqi": us_aqi,
        "temperature_2m": temp,
        "relative_humidity_2m": humidity,
        "pressure_msl": pressure,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir,
        "precipitation": precip,
    })
    return df


def populate_demo_feature_store(cities: list, days: int = 365):
    fs = FeatureStore()
    for city in cities:
        hourly = generate_hourly(city, days=days)
        features = engineer_features(hourly)
        n = fs.write(city, features)
        print(f"[demo_data] {city}: generated {len(features)} rows (feature store: {n} rows)")


if __name__ == "__main__":
    populate_demo_feature_store(list(CITIES.keys()), days=365)
