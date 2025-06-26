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

SEQUENCE_LENGTH = 30
TARGET_COLUMN = "PM2.5"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def create_sequences(df, feature_cols, target_col, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq_x = df[feature_cols].iloc[i:i + sequence_length].values
        seq_y = df[target_col].iloc[i + sequence_length]
        if not np.isnan(seq_x).any() and not np.isnan(seq_y):
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

def main():
    pipeline_dir = get_latest_pipeline_dir()
    input_path = os.path.join(pipeline_dir, "step6_scaled_data.csv")
    output_dir = os.path.join(pipeline_dir, "prepared_dataset")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.sort_values("date")

    with open(os.path.join(pipeline_dir, "selected_features.json"), "r") as f:
        feature_cols = json.load(f)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data.")

    logging.info(f"Using features: {feature_cols}")
    logging.info(f"Target: {TARGET_COLUMN}")

    X, y = create_sequences(df, feature_cols, TARGET_COLUMN, SEQUENCE_LENGTH)

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
