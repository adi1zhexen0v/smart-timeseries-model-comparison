import os
import sys
import argparse
import logging
from utils import get_project_root, get_latest_pipeline_dir

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.training.lstm_trainer import train_lstm
from src.visualization.loss_plotter import plot_loss
from src.visualization.real_vs_predicted_plotter import plot_real_vs_predicted

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Train LSTM model on time-series data.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    pipeline_dir = get_latest_pipeline_dir()
    dataset_dir = os.path.join(pipeline_dir, "prepared_dataset")

    model_path = os.path.join(project_root, "outputs", "models", "lstm_model.keras")
    loss_plot_path = os.path.join(project_root, "outputs", "diagrams", "lstm_loss.png")
    real_vs_predicted_plot_path = os.path.join(project_root, "outputs", "diagrams", "real_vs_predicted.png")
    scaler_path = os.path.join(pipeline_dir, "scaler_params.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)

    logging.info(f"Training LSTM model from {dataset_dir}")
    model, history, data = train_lstm(
        dataset_dir=dataset_dir,
        save_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    logging.info(f"Model saved to {model_path}")
    plot_loss(history, loss_plot_path)
    logging.info(f"Loss plot saved to {loss_plot_path}")

    plot_real_vs_predicted(dataset_dir, model_path, scaler_path, real_vs_predicted_plot_path)
    logging.info(f"Real vs Predicted plot saved to {real_vs_predicted_plot_path}")

    test_loss, test_mae = model.evaluate(data["X_test"], data["y_test"])
    logging.info(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
