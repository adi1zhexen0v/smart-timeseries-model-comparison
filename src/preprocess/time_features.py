import pandas as pd

def set_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    return "unknown"

def add_time_features(df: pd.DataFrame, key: str = "date") -> pd.DataFrame:
    df[key] = pd.to_datetime(df[key])

    df["dayofweek"] = df[key].dt.dayofweek
    df["month"] = df[key].dt.month
    df["day"] = df[key].dt.day

    df["season"] = df["month"].apply(set_season)
    df["season"] = df["season"].map({"winter": 0, "spring": 1, "summer": 2, "autumn": 3})
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    return df
