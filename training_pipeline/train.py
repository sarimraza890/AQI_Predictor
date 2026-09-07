import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITIES, DEFAULT_CITY, REPORTS_DIR
from feature_pipeline.features import FEATURE_COLUMNS, TARGET_COLUMNS
from feature_store.store import FeatureStore
from model_registry.registry import ModelRegistry

import shap

def get_model_zoo():
    return {
        "ridge_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42,
        ),
        "neural_net_mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", max_iter=600,
                early_stopping=True, random_state=42,
            )),
        ]),
    }


def load_training_frame(city: str) -> pd.DataFrame:
    fs = FeatureStore()
    df = fs.read(city)
    if df.empty:
        raise RuntimeError(f"No feature store data found for {city}. Run feature pipeline first.")
    return df


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def generate_shap_plot(model, X_test, city, horizon, model_name):
    try:
        sample = X_test.sample(min(150, len(X_test)), random_state=42)
        if model_name in ("random_forest", "gradient_boosting"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
            scaler = model.named_steps["scaler"] if hasattr(model, "named_steps") else None
            background = sample.sample(min(40, len(sample)), random_state=42)
            bg = scaler.transform(background) if scaler is not None else background
            sm = scaler.transform(sample) if scaler is not None else sample
            explainer = shap.KernelExplainer(inner.predict, bg)
            shap_values = explainer.shap_values(sm, nsamples=80)

        plt.figure(figsize=(9, 5))
        shap.summary_plot(shap_values, sample, show=False, max_display=12)
        out_path = os.path.join(REPORTS_DIR, f"shap_{city.lower()}_{horizon}.png")
        plt.title(f"{city.capitalize()} - {horizon.upper()} SHAP Feature Importance", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
    except Exception as e:
        print(f"  [{horizon}] SHAP generation notice: {e}")


def train_for_horizon(df: pd.DataFrame, target_col: str, city: str, horizon: str, registry: ModelRegistry):
    data = df.dropna(subset=FEATURE_COLUMNS + [target_col]).copy()
    X = data[FEATURE_COLUMNS]
    y = data[target_col]

    split_idx = int(len(data) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = get_model_zoo()
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        trained_models[name] = model
        registry.save(city, horizon, model, metrics, FEATURE_COLUMNS, name)
        print(f"  [{horizon}] {name:<18} RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    best_name = registry.promote_best(city, horizon)
    print(f"  [{horizon}] Promoted '{best_name}' to production")

    best_model = trained_models[best_name]
    generate_shap_plot(best_model, X_test, city, horizon, best_name)

    return results, best_name


def run(city: str = DEFAULT_CITY):
    if city not in CITIES:
        raise ValueError(f"City '{city}' not recognized. Must be one of {list(CITIES.keys())}")

    df = load_training_frame(city)
    registry = ModelRegistry()
    all_results = {}

    for target_col, horizon in zip(TARGET_COLUMNS, ["day1", "day2", "day3"]):
        print(f"\n--- Training {city} Models for {horizon.upper()} Horizon ---")
        results, best = train_for_horizon(df, target_col, city, horizon, registry)
        all_results[horizon] = {"results": results, "best_model": best}

    summary_path = os.path.join(REPORTS_DIR, f"training_summary_{city.lower()}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[training_pipeline] Summary successfully written to {summary_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Training Pipeline")
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES.keys()))
    parser.add_argument("--all-cities", action="store_true")
    args = parser.parse_args()

    target_cities = list(CITIES.keys()) if args.all_cities else [args.city]
    for c in target_cities:
        run(c)
