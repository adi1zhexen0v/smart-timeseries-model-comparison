import os
import sys
import logging
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.extract.sensor_loader import collect_all_stations_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    df = collect_all_stations_data()

    if not df.empty:
        processed_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)

        df["date"] = pd.to_datetime(df["date"])

        filename = f"combined_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        output_path = os.path.join(processed_dir, filename)

        df.to_csv(output_path, index=False)
        logging.info(f"Saved combined data to {output_path}")
    else:
        logging.warning("No data to save.")
