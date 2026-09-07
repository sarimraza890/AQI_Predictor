import sys
import os
import argparse
import pandas as pd
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, DEFAULT_CITY
from feature_pipeline.connectors import OpenMeteoConnector
from feature_pipeline.features import build_hourly_frame, engineer_features
from feature_store.store import FeatureStore


def run(city: str = DEFAULT_CITY, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    if city not in CITIES:
        raise ValueError(f"City '{city}' is not in supported list: {list(CITIES.keys())}")

    coords = CITIES[city]
    connector = OpenMeteoConnector()

    pollutants = connector.fetch_pollutants(coords["lat"], coords["lon"], start_date, end_date)
    weather = connector.fetch_weather(coords["lat"], coords["lon"], start_date, end_date)

    hourly = build_hourly_frame(pollutants, weather)
    features = engineer_features(hourly)

    fs = FeatureStore()
    total_rows = fs.write(city, features)
    print(f"[feature_pipeline] {city}: successfully processed {len(features)} rows (total in store: {total_rows})")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Ingestion Pipeline")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES.keys()), help="Target city")
    parser.add_argument("--start-date", default=None, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Backfill end date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(args.city, args.start_date, args.end_date)
