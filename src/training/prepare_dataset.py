import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_sequences(df, feature_cols, target_col, sequence_length):
    X, y = [], []

    if "site_id" in df.columns:
        grouped = df.groupby("site_id")
    else:
        grouped = [("all", df)]

    for _, group in grouped:
        group = group.sort_values("date")
        for i in range(len(group) - sequence_length):
            seq_x = group[feature_cols].iloc[i:i + sequence_length].values
            seq_y = group[target_col].iloc[i + sequence_length]
            if not np.isnan(seq_x).any() and not np.isnan(seq_y):
                X.append(seq_x)
                y.append(seq_y)

    return np.array(X), np.array(y)


def manual_split(X, y, train_frac=0.6, val_frac=0.2):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def prepare_dataset(input_path, features_path, output_dir, target_column="PM2.5", sequence_length=30, dataset_type=None):
    df = pd.read_csv(input_path)

    if "date" not in df.columns:
        raise ValueError("Expected 'date' column not found in dataset.")

    df["date"] = pd.to_datetime(df["date"])

    # Сортировка для временной последовательности (по site_id, если есть)
    if dataset_type == "traffic" and "site_id" in df.columns:
        df = df.sort_values(by=["site_id", "date"])
    else:
        df = df.sort_values("date")

    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    logging.info(f"Creating sequences with target '{target_column}' and length {sequence_length}")
    X, y = create_sequences(df, feature_cols, target_column, sequence_length)

    dataset = manual_split(X, y)

    os.makedirs(output_dir, exist_ok=True)
    for split_name, arr in dataset.items():
        np.save(os.path.join(output_dir, f"{split_name}.npy"), arr)
        logging.info(f"Saved {split_name}.npy to {output_dir}")
