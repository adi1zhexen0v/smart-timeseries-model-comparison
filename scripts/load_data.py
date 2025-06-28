import os
import sys
import pandas as pd
import logging
from datetime import datetime
from utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.domains.air_pollution.sensor_loader import collect_all_stations_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    df = collect_all_stations_data()

    if not df.empty:
        df  = df[(df["PM2.5"] >= 0) & (df["PM2.5"] <= 300)]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(project_root, "data", "processed", timestamp)
        os.makedirs(output_dir, exist_ok=True)

        latest_path_file = os.path.join(project_root, "data", "processed", "latest.txt")
        with open(latest_path_file, "w") as f:
            f.write(output_dir)

        df["date"] = pd.to_datetime(df["date"])
        combined_path = os.path.join(output_dir, "step1_combined_raw.csv")
        df.to_csv(combined_path, index=False)

        logging.info(f"Saved combined data to {combined_path}")
        logging.info(f"Pipeline root set to {output_dir}")
    else:
        logging.warning("No data to save.")

if __name__ == "__main__":
    main()
