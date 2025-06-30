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

from src.training.tcn_trainer import train_tcn
from src.visualization.loss_plotter import plot_loss
from src.visualization.real_vs_predicted_plotter import plot_real_vs_predicted

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TARGET_COLUMNS = {
    "air_pollution": "PM2.5",
    "traffic": "flow",
    "energy": "consumption"
}

def parse_dilations(d_str):
    return tuple(int(d) for d in d_str.split(","))

def main():
    parser = argparse.ArgumentParser(description="Train TCN model on time-series data.")
    parser.add_argument("--dataset_type", type=str, default="air_pollution", help="Dataset type: air_pollution, traffic, energy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nb_filters", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dilations", type=str, default="1,2,4,8", help="Comma-separated list of dilation values")
    parser.add_argument("--nb_stacks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense_units", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="tcn_default", help="Model tag or variant name")
    args = parser.parse_args()

    dataset_dir = os.path.join(project_root, "data", args.dataset_type, "processed")
    scaler_path = os.path.join(dataset_dir, "scaler_params.json")

    target_col = TARGET_COLUMNS.get(args.dataset_type)
    if target_col is None:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    model_subdir = os.path.join("outputs", "models", args.dataset_type)
    diagram_subdir = os.path.join("outputs", "diagrams", args.dataset_type)
    metrics_subdir = os.path.join("outputs", "metrics", args.dataset_type)
    os.makedirs(os.path.join(project_root, model_subdir), exist_ok=True)
    os.makedirs(os.path.join(project_root, diagram_subdir), exist_ok=True)
    os.makedirs(os.path.join(project_root, metrics_subdir), exist_ok=True)

    model_filename = f"{args.dataset_type}_{args.model_name}_tcn.keras"
    model_path = os.path.join(project_root, model_subdir, model_filename)

    logging.info("Training TCN model with parameters:")
    logging.info(f"filters={args.nb_filters}, kernel_size={args.kernel_size}, "
                 f"dilations={args.dilations}, stacks={args.nb_stacks}, dropout={args.dropout}, "
                 f"dense_units={args.dense_units}")

    start_time = time.time()
    model, history, data = train_tcn(
        dataset_dir=dataset_dir,
        output_dir=os.path.join(project_root, model_subdir),
        dataset_type=args.dataset_type,
        tag=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        nb_filters=args.nb_filters,
        kernel_size=args.kernel_size,
        dilations=parse_dilations(args.dilations),
        nb_stacks=args.nb_stacks,
        dropout=args.dropout,
        dense_units=args.dense_units,
        model_path=model_path,
        target_column=target_col
    )
    end_time = time.time()
    training_time = end_time - start_time

    plot_loss(
        history=history,
        model_name=args.model_name,
        model_type="tcn",
        output_dir=os.path.join(project_root, diagram_subdir)
    )

    plot_real_vs_predicted(
        dataset_dir=dataset_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        model_name=args.model_name,
        model_type="tcn",
        target_col=target_col,
        output_dir=os.path.join(project_root, diagram_subdir)
    )

    test_loss, test_mae = model.evaluate(data["X_test"], data["y_test"], verbose=1)
    y_pred = model.predict(data["X_test"]).flatten()
    r2 = r2_score(data["y_test"], y_pred)

    logging.info(f"Test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}, R²: {r2:.4f}")
    logging.info(f"Training time: {training_time:.2f} seconds")

    metrics = {
        "mse": float(test_loss),
        "mae": float(test_mae),
        "r2": float(r2),
        "training_time_seconds": round(training_time, 2),
        "nb_filters": args.nb_filters,
        "kernel_size": args.kernel_size,
        "dilations": parse_dilations(args.dilations),
        "nb_stacks": args.nb_stacks,
        "dropout": args.dropout,
        "dense_units": args.dense_units,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    metrics_path = os.path.join(project_root, metrics_subdir, f"{args.model_name}_tcn_metrics.json")
    history_path = os.path.join(project_root, metrics_subdir, f"{args.model_name}_tcn_history.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

if __name__ == "__main__":
    main()
