import os
import sys
import json
import time
import argparse
import logging
from sklearn.metrics import r2_score
from utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.training.lstm_trainer import train_lstm
from src.visualization.loss_plotter import plot_loss
from src.visualization.real_vs_predicted_plotter import plot_real_vs_predicted

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TARGET_COLUMNS = {
    "air_pollution": "PM2.5",
    "traffic": "flow",
    "energy": "consumption"
}

def main():
    parser = argparse.ArgumentParser(description="Train LSTM model on time-series data.")
    parser.add_argument("--dataset_type", type=str, default="air_pollution", help="Dataset type: air_pollution, traffic, energy")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--units", type=int, default=64, help="Number of LSTM units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate after LSTM")
    parser.add_argument("--dense_units", type=int, default=32, help="Number of units in Dense layer")
    parser.add_argument("--model_name", type=str, default="lstm_default", help="Model tag or variant name")

    args = parser.parse_args()

    target_col = TARGET_COLUMNS.get(args.dataset_type)
    if target_col is None:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    dataset_dir = os.path.join(project_root, "data", args.dataset_type, "processed")
    scaler_path = os.path.join(project_root, "data", args.dataset_type, "processed", "scaler_params.json")

    model_subdir = os.path.join("outputs", "models", args.dataset_type)
    diagram_subdir = os.path.join("outputs", "diagrams", args.dataset_type)
    metrics_subdir = os.path.join("outputs", "metrics", args.dataset_type)
    os.makedirs(os.path.join(project_root, model_subdir), exist_ok=True)
    os.makedirs(os.path.join(project_root, diagram_subdir), exist_ok=True)
    os.makedirs(os.path.join(project_root, metrics_subdir), exist_ok=True)

    model_path = os.path.join(project_root, model_subdir, f"{args.dataset_type}_{args.model_name}_lstm.keras")

    logging.info(f"Training LSTM model from {dataset_dir}")
    logging.info(f"Parameters: units={args.units}, dropout={args.dropout}, dense_units={args.dense_units}")

    start_time = time.time()
    model, history, data = train_lstm(
        dataset_dir=dataset_dir,
        output_dir=os.path.join(project_root, model_subdir),
        dataset_type=args.dataset_type,
        tag=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dropout=args.dropout,
        dense_units=args.dense_units,
        target_column=target_col
    )
    end_time = time.time()
    training_time = end_time - start_time
    epoch_time = training_time / args.epochs

    plot_loss(
        history=history,
        model_name=args.model_name,
        model_type="lstm",
        output_dir=os.path.join(project_root, diagram_subdir)
    )

    plot_real_vs_predicted(
        dataset_dir=dataset_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        model_name=args.model_name,
        model_type="lstm",
        target_col=target_col,
        output_dir=os.path.join(project_root, diagram_subdir)
    )

    test_loss, test_mae = model.evaluate(data["X_test"], data["y_test"], verbose=1)
    y_pred = model.predict(data["X_test"]).flatten()
    r2 = r2_score(data["y_test"], y_pred)

    logging.info(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}, R²: {r2:.4f}")
    logging.info(f"Training time: {training_time:.2f} seconds | Epoch time: {epoch_time:.2f} seconds")

    metrics = {
        "mse": float(test_loss),
        "mae": float(test_mae),
        "r2": float(r2),
        "training_time_seconds": round(training_time, 2),
        "epoch_time_seconds": round(epoch_time, 2),
        "units": args.units,
        "dropout": args.dropout,
        "dense_units": args.dense_units,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    metrics_path = os.path.join(project_root, metrics_subdir, f"{args.model_name}_lstm_metrics.json")
    history_path = os.path.join(project_root, metrics_subdir, f"{args.model_name}_lstm_history.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

if __name__ == "__main__":
    main()
