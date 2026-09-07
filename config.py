import os

CITIES = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "province": "Sindh"},
    "Lahore": {"lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "province": "Punjab"},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "province": "Federal"},
}
DEFAULT_CITY = "Karachi"

OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENWEATHER_AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AQICN_FEED_URL = "https://api.waqi.info/feed"

POLLUTANT_HOURLY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
]
WEATHER_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store", "data")
MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory and poses little or no risk."),
    (51, 100, "Moderate", "#ffff00", "Acceptable air quality; moderate health concern for a very small number of unusually sensitive individuals."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects. General public is less likely to be affected."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may begin to experience health effects; sensitive group members may experience more serious health effects."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: risk of health effects is increased for everyone."),
    (301, 500, "Hazardous", "#7e0023", "Health warning of emergency conditions: entire population is likely to be affected."),
]

HAZARD_ALERT_THRESHOLD = 150

def aqi_category(aqi_value: float):
    if aqi_value is None:
        return ("Unknown", "#888888", "No data available")
    for lo, hi, label, color, advice in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return (label, color, advice)
    return ("Hazardous", "#7e0023", AQI_CATEGORIES[-1][4])
