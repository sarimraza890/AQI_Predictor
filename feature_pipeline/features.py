import numpy as np
import pandas as pd


def build_hourly_frame(pollutant_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(pollutant_df, weather_df, on="time", how="inner")
    df = df.rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    df["hour"] = df["ts"].dt.hour
    df["day"] = df["ts"].dt.day
    df["month"] = df["ts"].dt.month
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["aqi_change_rate_1h"] = df["us_aqi"].diff()
    df["aqi_change_rate_3h"] = df["us_aqi"].diff(3)
    df["aqi_rolling_mean_24h"] = df["us_aqi"].rolling(24, min_periods=1).mean()
    df["aqi_rolling_std_24h"] = df["us_aqi"].rolling(24, min_periods=1).std()
    df["aqi_rolling_max_24h"] = df["us_aqi"].rolling(24, min_periods=1).max()
    df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
    df["pollutant_load"] = (
        df["pm2_5"].fillna(0) + df["pm10"].fillna(0) +
        df["nitrogen_dioxide"].fillna(0) + df["sulphur_dioxide"].fillna(0) +
        df["ozone"].fillna(0) + df["carbon_monoxide"].fillna(0) / 100
    )

    df["is_spike"] = (
        (df["aqi_change_rate_3h"].abs() > 2 * df["aqi_rolling_std_24h"].fillna(0).clip(lower=1))
    ).astype(int)

    for lag in [1, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
        df[f"temp_lag_{lag}h"] = df["temperature_2m"].shift(lag)

    daily_max_aqi = df.set_index("ts")["us_aqi"].resample("1D").max()
    df["date"] = df["ts"].dt.floor("D")
    for h in [1, 2, 3]:
        shifted = daily_max_aqi.shift(-h)
        df[f"target_day{h}_aqi"] = df["date"].map(shifted)

    df = df.drop(columns=["date"])
    return df


FEATURE_COLUMNS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m",
    "wind_direction_10m", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
    "aqi_change_rate_1h", "aqi_change_rate_3h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h", "aqi_rolling_max_24h",
    "pm25_pm10_ratio", "pollutant_load", "is_spike",
    "aqi_lag_1h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "temp_lag_1h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
]

TARGET_COLUMNS = ["target_day1_aqi", "target_day2_aqi", "target_day3_aqi"]
