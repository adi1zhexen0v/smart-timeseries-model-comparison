import os
import sys
import logging
import pandas as pd
from utils import get_project_root, get_latest_pipeline_dir

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.preprocess.fetch_weather_data import add_weather_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step1_combined_raw.csv")
    output_path = os.path.join(pipeline_dir, "step2_weather_merged.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"combined.csv not found in {pipeline_dir}")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    logging.info(f"Loaded combined data from {input_path}")
    logging.info("Fetching weather data from Open-Meteo...")

    enriched_df = add_weather_columns(df)
    enriched_df.to_csv(output_path, index=False)
    logging.info(f"Saved weather-augmented dataset to {output_path}")

if __name__ == "__main__":
    main()
