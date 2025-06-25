import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def filter_stations_by_history(df: pd.DataFrame, min_days: int = 730) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])

    station_ranges = df.groupby("station_name")["date"].agg(["min", "max"])
    station_ranges["duration_days"] = (station_ranges["max"] - station_ranges["min"]).dt.days

    valid_stations = station_ranges[station_ranges["duration_days"] >= min_days].index
    filtered_df = df[df["station_name"].isin(valid_stations)].copy()

    logging.info(f"Kept {len(valid_stations)} stations with ≥ {min_days} days of data.")
    return filtered_df
