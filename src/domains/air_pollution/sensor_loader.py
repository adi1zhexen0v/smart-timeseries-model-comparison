import os
import logging
import pandas as pd
from typing import Optional
from data.air_pollution.stations import stations

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_single_parameter(file_path: str, param_name: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path, on_bad_lines="skip")
    except Exception as e:
        logging.warning(f"Error while reading file {file_path}: {e}")
        return None

    if "date" in df.columns and "median" in df.columns:
        df = df[["date", "median"]].copy()
        df.rename(columns={"median": param_name}, inplace=True)
        return df
    else:
        logging.info(f"File '{file_path}' skipped: missing necessary columns")
        return None


def load_station_data(folder_path: str, station_name: str, lat: float, lon: float) -> pd.DataFrame:
    parameter_frames = {}

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file)
        if os.path.getsize(file_path) == 0:
            logging.info(f"Skipped empty file: {file_path}")
            continue

        param_name = os.path.splitext(file)[0].split("_")[-1]
        df = load_single_parameter(file_path, param_name)

        if df is not None:
            parameter_frames[param_name] = df

    if not parameter_frames:
        logging.warning(f"No valid parameters found for station {station_name}")
        return pd.DataFrame()

    merged_df = None
    for param_name, param_df in parameter_frames.items():
        merged_df = param_df if merged_df is None else pd.merge(merged_df, param_df, on="date", how="outer")

    merged_df["station_name"] = station_name
    merged_df["latitude"] = lat
    merged_df["longitude"] = lon
    merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.date
    merged_df = merged_df.drop_duplicates(subset=["date", "station_name"])
    merged_df = merged_df.sort_values("date")

    return merged_df


def collect_all_stations_data(base_folder: str) -> pd.DataFrame:
    all_data = []

    for station in stations:
        folder_path = os.path.join(base_folder, station["folder"])
        df = load_station_data(
            folder_path=folder_path,
            station_name=station["name"],
            lat=station["lat"],
            lon=station["lon"]
        )
        if not df.empty:
            logging.info(f"Collected data for station: {station['name']}")
            all_data.append(df)
        else:
            logging.warning(f"No data collected for station: {station['name']}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logging.info(f"Collected data from {len(all_data)} stations")
        return combined
    else:
        logging.warning("No data collected from any station")
        return pd.DataFrame()

