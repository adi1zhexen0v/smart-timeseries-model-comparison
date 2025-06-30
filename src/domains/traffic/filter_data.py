import pandas as pd

def filter_and_sort_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"timestamp": "date"})
    df = df[(df["lat"] != 0.0) & (df["long"] != 0.0)]
    df = df.drop_duplicates(subset=["date", "site_id"])
    df = df.sort_values(by=["date", "site_id"]).reset_index(drop=True)
    return df
