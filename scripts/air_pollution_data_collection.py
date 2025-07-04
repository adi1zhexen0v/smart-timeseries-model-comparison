import os
import sys
import json
import logging
from json import load
from utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.domains.air_pollution.sensor_loader import collect_all_stations_data
from src.domains.air_pollution.fetch_weather_data import add_weather_columns
from src.domains.air_pollution.filter_stations import filter_stations_by_history
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import scale_features
from src.preprocess.time_features import add_time_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    project_root = get_project_root()

    raw_dir = os.path.join(project_root, "data", "air_pollution", "raw")
    processed_dir = os.path.join(project_root, "data", "air_pollution", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = collect_all_stations_data(raw_dir)

    df = add_weather_columns(df)
    df = filter_stations_by_history(df, min_days=730)
    df = select_features(df, processed_dir, "air_pollution", threshold=0.9)
    df = add_time_features(df)

    selected_features_path = os.path.join(processed_dir, "selected_features.json")
    with open(selected_features_path, "r") as f:
        selected_features = load(f)

    df_scaled, scaler = scale_features(df, selected_features, target="PM2.5")

    scaler_serializable = {
        k: {ik: float(iv) for ik, iv in v.items()}
        for k, v in scaler.items()
    }

    df_scaled.to_csv(os.path.join(processed_dir, "air_pollution_dataset.csv"), index=False)
    with open(os.path.join(processed_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_serializable, f, indent=2)

    logging.info("Pipeline complete: processed data saved.")
