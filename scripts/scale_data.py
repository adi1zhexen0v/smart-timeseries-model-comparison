import os
import sys
import json
import pandas as pd
import logging
from utils import get_project_root, get_latest_pipeline_dir

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.preprocess.scaler import scale_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step5_time_features.csv")
    output_path = os.path.join(pipeline_dir, "step6_scaled_data.csv")
    scaler_path = os.path.join(pipeline_dir, "scaler_params.json")
    features_path = os.path.join(pipeline_dir, "selected_features.json")

    if not os.path.exists(input_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Required input or feature list is missing.")

    df = pd.read_csv(input_path)
    with open(features_path, "r") as f:
        selected_features = json.load(f)

    logging.info(f"Loaded dataset from {input_path}")
    df_scaled, scaler_params = scale_features(df, selected_features)

    df_scaled.to_csv(output_path, index=False)
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)

    logging.info(f"Saved scaled dataset to {output_path}")
    logging.info(f"Saved scaler parameters to {scaler_path}")


if __name__ == "__main__":
    main()
