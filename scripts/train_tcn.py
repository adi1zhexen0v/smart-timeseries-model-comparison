import os
import sys
import json
import time
import argparse
import logging
from utils import get_project_root, get_latest_pipeline_dir
from sklearn.metrics import r2_score

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.training.tcn_trainer import train_tcn
from src.visualization.loss_plotter import plot_loss
from src.visualization.real_vs_predicted_plotter import plot_real_vs_predicted

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_dilations(d_str):
    return tuple(int(d) for d in d_str.split(","))

def main():
    parser = argparse.ArgumentParser(description="Train TCN model on time-series data.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nb_filters", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dilations", type=str, default="1,2,4,8", help="Comma-separated list of dilation values")
    parser.add_argument("--nb_stacks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense_units", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="tcn_default", help="Prefix for saving model and plots")
    args = parser.parse_args()

    dilations = parse_dilations(args.dilations)

    pipeline_dir = get_latest_pipeline_dir()
    dataset_dir = os.path.join(pipeline_dir, "prepared_dataset")

    model_path = os.path.join(project_root, "outputs", "models", f"{args.model_name}_model.keras")
    diagrams_dir = os.path.join(project_root, "outputs", "diagrams")
    scaler_path = os.path.join(pipeline_dir, "scaler_params.json")
    metrics_dir = os.path.join(project_root, "outputs", "metrics")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(diagrams_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logging.info("Training TCN model with parameters:")
    logging.info(f"filters={args.nb_filters}, kernel_size={args.kernel_size}, "
                 f"dilations={dilations}, stacks={args.nb_stacks}, dropout={args.dropout}, "
                 f"dense_units={args.dense_units}")

    start_time = time.time()
    model, history, data = train_tcn(
        dataset_dir=dataset_dir,
        save_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        nb_filters=args.nb_filters,
        kernel_size=args.kernel_size,
        dilations=dilations,
        nb_stacks=args.nb_stacks,
        dropout=args.dropout,
        dense_units=args.dense_units
    )
    end_time = time.time()
    training_time = end_time - start_time

    plot_loss(history, args.model_name, diagrams_dir)
    plot_real_vs_predicted(
        dataset_dir=dataset_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        model_name=args.model_name,
        output_dir=diagrams_dir
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
        "training_time_seconds": float(training_time),
        "nb_filters": args.nb_filters,
        "kernel_size": args.kernel_size,
        "dilations": dilations,
        "nb_stacks": args.nb_stacks,
        "dropout": args.dropout,
        "dense_units": args.dense_units,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    with open(os.path.join(metrics_dir, f"{args.model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
