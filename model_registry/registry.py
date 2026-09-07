import os
import json
import joblib
from datetime import datetime, timezone
from config import MODEL_REGISTRY_DIR


class ModelRegistry:
    def __init__(self, base_dir: str = MODEL_REGISTRY_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, city: str, horizon: str, model, metrics: dict, feature_names: list, model_name: str):
        safe_city = city.lower().replace(" ", "_")
        folder = os.path.join(self.base_dir, safe_city, horizon)
        os.makedirs(folder, exist_ok=True)

        model_path = os.path.join(folder, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        meta_path = os.path.join(folder, f"{model_name}.json")
        meta = {
            "city": city,
            "horizon": horizon,
            "model_name": model_name,
            "metrics": metrics,
            "feature_names": feature_names,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return model_path

    def promote_best(self, city: str, horizon: str):
        safe_city = city.lower().replace(" ", "_")
        folder = os.path.join(self.base_dir, safe_city, horizon)
        if not os.path.isdir(folder):
            return None
        best_meta, best_name = None, None
        for fname in os.listdir(folder):
            if fname.endswith(".json") and fname != "production.json":
                with open(os.path.join(folder, fname)) as f:
                    meta = json.load(f)
                if best_meta is None or meta["metrics"]["rmse"] < best_meta["metrics"]["rmse"]:
                    best_meta = meta
                    best_name = meta["model_name"]
        if best_name:
            with open(os.path.join(folder, "production.json"), "w") as f:
                json.dump({"model_name": best_name}, f)
        return best_name

    def load_production(self, city: str, horizon: str):
        safe_city = city.lower().replace(" ", "_")
        folder = os.path.join(self.base_dir, safe_city, horizon)
        ptr_path = os.path.join(folder, "production.json")
        if not os.path.exists(ptr_path):
            return None, None
        with open(ptr_path) as f:
            model_name = json.load(f)["model_name"]
        return self.load_model(city, horizon, model_name)

    def load_model(self, city: str, horizon: str, model_name: str):
        safe_city = city.lower().replace(" ", "_")
        folder = os.path.join(self.base_dir, safe_city, horizon)
        model_path = os.path.join(folder, f"{model_name}.joblib")
        meta_path = os.path.join(folder, f"{model_name}.json")
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return None, None
        model = joblib.load(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return model, meta

    def list_models(self, city: str, horizon: str):
        safe_city = city.lower().replace(" ", "_")
        folder = os.path.join(self.base_dir, safe_city, horizon)
        if not os.path.isdir(folder):
            return []
        models = []
        for fname in os.listdir(folder):
            if fname.endswith(".json") and fname != "production.json":
                with open(os.path.join(folder, fname)) as f:
                    models.append(json.load(f))
        return sorted(models, key=lambda m: m["metrics"]["rmse"])
