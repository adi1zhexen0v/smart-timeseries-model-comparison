import os
import sys
import logging
import pandas as pd
from utils import get_project_root, get_latest_pipeline_dir

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.preprocess.feature_selector import select_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step3_filtered_min_days.csv")
    output_path = os.path.join(pipeline_dir, "step4_features_selected.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run station_filter.py first.")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    logging.info(f"Loaded filtered dataset from {input_path}")
    selected_df = select_features(df, output_dir=pipeline_dir, threshold=0.9)

    selected_df.to_csv(output_path, index=False)
    logging.info(f"Saved feature-selected dataset to {output_path}")


if __name__ == "__main__":
    main()
