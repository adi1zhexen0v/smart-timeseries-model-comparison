import os
import sys
import json
import logging
from json import load
from utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.domains.traffic.merge_csv_files import merge_traffic_csv_files
from src.domains.traffic.filter_data import filter_and_sort_traffic_data
from src.domains.traffic.fill_missing_data import fill_missing_traffic_data
from src.preprocess.feature_selector import select_features
from src.preprocess.time_features import add_time_features
from src.preprocess.scaler import scale_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    input_dir = os.path.join(project_root, "data", "traffic", "raw")
    output_dir = os.path.join(project_root, "data", "traffic", "processed")
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Merging raw Glasgow traffic CSV files...")
    merged_df = merge_traffic_csv_files(input_dir)

    logging.info("Filtering and sorting merged traffic data...")
    cleaned_df = filter_and_sort_traffic_data(merged_df)
    cleaned_df.to_csv(os.path.join(output_dir, "cleaned_traffic_dataset.csv"), index=False)

    logging.info("Filling missing records...")
    filled_df = fill_missing_traffic_data(cleaned_df)

    logging.info("Selecting features...")
    selected_df = select_features(filled_df, output_dir, dataset_type="traffic", threshold=0.9)

    logging.info("Adding time features...")
    enriched_df = add_time_features(selected_df)

    logging.info("Scaling features...")
    selected_features_path = os.path.join(output_dir, "selected_features.json")
    with open(selected_features_path, "r") as f:
        selected_features = load(f)

    df_scaled, scaler = scale_features(enriched_df, selected_features, target=None)

    scaler_serializable = {
        k: {ik: float(iv) for ik, iv in v.items()}
        for k, v in scaler.items()
    }

    df_scaled.to_csv(os.path.join(output_dir, "traffic_dataset.csv"), index=False)
    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_serializable, f, indent=2)

    logging.info("Pipeline complete: processed data saved.")
    logging.info(f"Final row count: {len(df_scaled)}")

if __name__ == "__main__":
    main()
