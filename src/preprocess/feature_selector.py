import os
import json
import pandas as pd
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


NON_FEATURE_COLUMNS = {
    "air_pollution": ["date", "station_name", "latitude", "longitude"],
    "traffic": ["date", "site_id", "lat", "long"],
    "energy": ["timestamp", "region", "latitude", "longitude"],
}


def select_features(df: pd.DataFrame, output_dir: str, dataset_type: str, threshold: float = 0.9) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type not in NON_FEATURE_COLUMNS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    non_feature_cols = NON_FEATURE_COLUMNS[dataset_type]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    completeness = {
        col: df[col].notnull().sum() / len(df) for col in feature_cols
    }

    selected_features: List[str] = [
        col for col, ratio in completeness.items() if ratio >= threshold
    ]

    logging.info(f"[{dataset_type}] Feature fill ratios:")
    for col in sorted(completeness, key=completeness.get, reverse=True):
        logging.info(f"  {col}: {completeness[col]*100:.2f}%")

    logging.info(f"[{dataset_type}] Selected features (≥ {int(threshold * 100)}%): {selected_features}")

    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    final_cols = non_feature_cols + selected_features
    filtered_df = df[final_cols].copy()
    filtered_df.dropna(subset=selected_features, how="all", inplace=True)

    return filtered_df
