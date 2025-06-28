import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def plot_real_vs_predicted(dataset_dir, model_path, scaler_path, dataset_type: str, model_type: str, target_col: str, output_dir: str):
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    model = load_model(model_path)
    y_pred = model.predict(X_test).flatten()

    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    mean = scaler[target_col]["mean"]
    std = scaler[target_col]["std"]

    y_test_denorm = y_test * std + mean
    y_pred_denorm = y_pred * std + mean

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_denorm[:200], label="Real", linewidth=2)
    plt.plot(y_pred_denorm[:200], label="Predicted", linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel(f"{target_col} (denormalized)")
    plt.title(f"Real vs Predicted {target_col} ({dataset_type.upper()} - {model_type.upper()})")
    plt.legend()
    plt.tight_layout()

    filename = f"{dataset_type}_{model_type}_real_vs_predicted.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    logging.info(f"Saved Real vs Predicted plot to {save_path}")