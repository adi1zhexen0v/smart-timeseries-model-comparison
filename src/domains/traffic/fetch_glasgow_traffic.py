import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "http://api.glasgow.gov.uk/traffic/v1/movement/history"
STEP = timedelta(days=7)
SLEEP_WEEKS = 60
SLEEP_PAGES = 5
PAGE_SIZE = 100
RETRY_LIMIT = 3
REQUEST_TIMEOUT = 90

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def fetch_glasgow_traffic(start_date: datetime, end_date: datetime, output_path: str):
    all_data = []
    current = start_date

    try:
        while current < end_date:
            week_start = current
            week_end = min(current + STEP, end_date)
            page = 1
            logging.info(f"Fetching data: {week_start.date()} to {week_end.date()}")

            while True:
                retry_count = 0
                while retry_count < RETRY_LIMIT:
                    try:
                        session = requests.Session()
                        session.headers.update({
                            "User-Agent": "Mozilla/5.0",
                            "Accept": "application/json",
                            "Connection": "keep-alive"
                        })

                        params = {
                            "page": page,
                            "size": PAGE_SIZE,
                            "format": "json",
                            "start": week_start.isoformat() + "Z",
                            "end": week_end.isoformat() + "Z"
                        }

                        response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                        response.raise_for_status()
                        data = response.json()

                        if not data:
                            logging.info(f"No more data for {week_start.date()}, page {page}")
                            break

                        all_data.extend(data)
                        logging.info(f"Retrieved {len(data)} records (page {page})")
                        page += 1
                        time.sleep(SLEEP_PAGES)
                        break

                    except requests.exceptions.ReadTimeout:
                        retry_count += 1
                        logging.warning(f"Read timeout on {week_start.date()}, page {page} (attempt {retry_count})")
                        time.sleep(10)

                    except requests.exceptions.RequestException as e:
                        retry_count += 1
                        logging.warning(f"Request error on {week_start.date()}, page {page} (attempt {retry_count}): {e}")
                        time.sleep(10)

                else:
                    logging.error(f"Failed to fetch page {page} for {week_start.date()} after {RETRY_LIMIT} attempts")
                    break

                if data == []:
                    break

            logging.info(f"Sleeping {SLEEP_WEEKS} seconds before the next batch...\n")
            time.sleep(SLEEP_WEEKS)
            current += STEP

    finally:
        if all_data:
            parsed = []
            for item in all_data:
                site = item.get("site", {})
                site_id = site.get("siteId", "")
                from_info = site.get("from", {})
                lat = from_info.get("lat", 0)
                lon = from_info.get("long", 0)

                parsed.append({
                    "timestamp": item.get("timestamp"),
                    "site_id": site_id,
                    "lat": lat,
                    "long": lon,
                    "flow": int(item.get("flow", 0)),
                    "concentration": item.get("concentration", 0)
                })

            df = pd.DataFrame(parsed)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["timestamp", "site_id"])

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved partial data to: {output_path}")
        else:
            logging.warning("No data was collected. Nothing to save.")
