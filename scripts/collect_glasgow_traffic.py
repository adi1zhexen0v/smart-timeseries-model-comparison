import os
import sys
import logging
from datetime import datetime
from utils import get_project_root

project_root = get_project_root()
sys.path.insert(0, project_root)

from src.domains.traffic.fetch_glasgow_traffic import fetch_glasgow_traffic

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    # start_date = datetime(2021, 9, 12)
    # end_date = datetime.utcnow() - timedelta(days=1)
    start_date = datetime(2021, 10, 31)
    end_date = datetime(2021, 12, 12)

    timestamp_id = int(datetime.utcnow().timestamp() * 1000)
    filename = f"{timestamp_id}_glasgow_traffic_data.csv"
    output_path = os.path.join("data", "traffic", "processed", filename)

    fetch_glasgow_traffic(start_date, end_date, output_path)
