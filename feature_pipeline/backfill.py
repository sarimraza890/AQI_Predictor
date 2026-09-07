import sys
import os
import argparse
from datetime import date, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, DEFAULT_CITY
from feature_pipeline.run import run


def backfill_city(city: str, days: int, chunk_days: int = 90):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    cursor = start_date

    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_date)
        print(f"[backfill] {city}: processing window {cursor} to {chunk_end}")
        run(city, start_date=cursor.isoformat(), end_date=chunk_end.isoformat())
        cursor = chunk_end + timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Feature Backfill")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES.keys()))
    parser.add_argument("--all-cities", action="store_true", help="Backfill Karachi, Lahore, and Islamabad")
    parser.add_argument("--days", type=int, default=365, help="Number of historical days")
    args = parser.parse_args()

    target_cities = list(CITIES.keys()) if args.all_cities else [args.city]
    for c in target_cities:
        backfill_city(c, args.days)
