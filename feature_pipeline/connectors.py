import os
import requests
import pandas as pd
from typing import Optional, Dict, Any
from config import (
    CITIES,
    OPEN_METEO_AIR_QUALITY_URL,
    OPEN_METEO_WEATHER_FORECAST_URL,
    OPEN_METEO_WEATHER_ARCHIVE_URL,
    OPENWEATHER_AIR_POLLUTION_URL,
    AQICN_FEED_URL,
    POLLUTANT_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
)

class OpenMeteoConnector:
    def fetch_pollutants(self, lat: float, lon: float, start_date: Optional[str] = None, end_date: Optional[str] = None, past_days: int = 7) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(POLLUTANT_HOURLY_VARS),
            "timezone": "UTC",
        }
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:
            params["past_days"] = past_days
            params["forecast_days"] = 1

        response = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=30)
        response.raise_for_status()
        return pd.DataFrame(response.json()["hourly"])

    def fetch_weather(self, lat: float, lon: float, start_date: Optional[str] = None, end_date: Optional[str] = None, past_days: int = 7) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_HOURLY_VARS),
            "timezone": "UTC",
        }
        if start_date and end_date:
            url = OPEN_METEO_WEATHER_ARCHIVE_URL
            params["start_date"] = start_date
            params["end_date"] = end_date
        else:
            url = OPEN_METEO_WEATHER_FORECAST_URL
            params["past_days"] = past_days
            params["forecast_days"] = 1

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return pd.DataFrame(response.json()["hourly"])


class OpenWeatherConnector:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY", "")

    def fetch_current(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        params = {"lat": lat, "lon": lon, "appid": self.api_key}
        response = requests.get(OPENWEATHER_AIR_POLLUTION_URL, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
        return None


class AQICNConnector:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("AQICN_API_TOKEN", "")

    def fetch_city_feed(self, city: str) -> Optional[Dict[str, Any]]:
        if not self.api_token:
            return None
        url = f"{AQICN_FEED_URL}/{city.lower()}/"
        params = {"token": self.api_token}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
        return None


def get_default_connector() -> OpenMeteoConnector:
    return OpenMeteoConnector()
