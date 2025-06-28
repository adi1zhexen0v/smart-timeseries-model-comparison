import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from scripts.utils import get_latest_pipeline_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_TARGET_COLUMN = "PM2.5"

def create_sequences(df, feature_cols, target_col, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq_x = df[feature_cols].iloc[i:i + sequence_length].values
        seq_y = df[target_col].iloc[i + sequence_length]
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step6_scaled_data.csv")
    feature_path = os.path.join(pipeline_dir, "selected_features.json")
    config_path = os.path.join(pipeline_dir, "prepare_config.json")
    output_dir = os.path.join(pipeline_dir, "prepared_dataset")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(config_path)
    target_col = config.get("target_column", DEFAULT_TARGET_COLUMN)
    sequence_length = config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)

    df = pd.read_csv(input_path)
    df = df.sort_values("date")

    with open(feature_path, "r") as f:
        feature_cols = json.load(f)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    logging.info(f"Using features: {feature_cols}")
    logging.info(f"Target: {target_col}")
    logging.info(f"Sequence length: {sequence_length}")

    X, y = create_sequences(df, feature_cols, target_col, sequence_length)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    logging.info(f"Saved prepared sequences to {output_dir}")

if __name__ == "__main__":
    main()
