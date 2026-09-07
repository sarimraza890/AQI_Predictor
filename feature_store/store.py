import os
import pandas as pd
from config import FEATURE_STORE_DIR


class FeatureStore:
    """
    Feature Store interface for ingestion and retrieval of time-series AQI features.
    Provides pluggable backend storage (Local Parquet, Hopsworks, or Vertex AI Feature Store).
    """

    def __init__(self, base_dir: str = FEATURE_STORE_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, city: str) -> str:
        safe = city.lower().replace(" ", "_")
        return os.path.join(self.base_dir, f"{safe}.parquet")

    def write(self, city: str, df: pd.DataFrame) -> int:
        path = self._path(city)
        df = df.copy()
        df["city"] = city
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ts"], keep="last")
        else:
            combined = df
        combined = combined.sort_values("ts").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        return len(combined)

    def read(self, city: str) -> pd.DataFrame:
        path = self._path(city)
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_all(self) -> pd.DataFrame:
        frames = []
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(self.base_dir, fname)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
