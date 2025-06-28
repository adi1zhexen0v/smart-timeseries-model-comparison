import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def filter_stations_by_history(df: pd.DataFrame, min_days: int = 730) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])

    station_counts = df.groupby("station_name")["date"].nunique()
    valid_stations = station_counts[station_counts >= min_days].index

    filtered_df = df[df["station_name"].isin(valid_stations)].copy()

    logging.info(f"Kept {len(valid_stations)} stations with ≥ {min_days} days after merge.")
    return filtered_df
