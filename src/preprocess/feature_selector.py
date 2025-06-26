import os
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def select_features(df: pd.DataFrame, output_dir: str, threshold: float = 0.9) -> pd.DataFrame:
    non_feature_cols = ["date", "station_name", "latitude", "longitude"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    completeness = {
        col: df[col].notnull().sum() / len(df) for col in feature_cols
    }

    selected_features = [col for col, ratio in completeness.items() if ratio >= threshold]

    logging.info("Feature fill ratios:")
    for col in sorted(completeness, key=completeness.get, reverse=True):
        logging.info(f"{col}: {completeness[col]*100:.2f}%")

    logging.info(f"Selected features (≥ {int(threshold * 100)}% filled): {selected_features}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    final_cols = non_feature_cols + selected_features
    filtered_df = df[final_cols].copy()

    filtered_df.dropna(subset=selected_features, how="all", inplace=True)

    return filtered_df
