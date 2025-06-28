import os
import sys
import logging
import pandas as pd
from utils import get_project_root, get_latest_pipeline_dir

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.domains.air_pollution.filter_stations import filter_stations_by_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step2_weather_merged.csv")
    output_path = os.path.join(pipeline_dir, "step3_filtered_min_days.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run append_weather.py first.")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    logging.info(f"Loaded weather-enriched data from {input_path}")
    filtered_df = filter_stations_by_history(df)

    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Saved filtered dataset to {output_path}")


if __name__ == "__main__":
    main()
