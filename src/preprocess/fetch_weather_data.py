import os
import time
import logging
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from typing import Tuple

# Setup cached session
cache = requests_cache.CachedSession(".cache", expire_after=-1)
session = retry(cache, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=session)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_hourly_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure"
        ],
        "timezone": "Asia/Almaty"
    }

    try:
        response = openmeteo.weather_api(url, params=params)[0]
        hourly = response.Hourly()

        df = pd.DataFrame({
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "pressure": hourly.Variables(3).ValuesAsNumpy()
        })

        df["datetime"] = df["datetime"].dt.tz_localize(None)
        df["date"] = df["datetime"].dt.date
        df.drop(columns=["datetime"], inplace=True)

        df["latitude"] = lat
        df["longitude"] = lon

        return df

    except Exception as e:
        logging.warning(f"Weather API failed for lat={lat}, lon={lon}: {e}")
        return pd.DataFrame()


def fetch_daily_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    hourly_df = fetch_hourly_weather(lat, lon, start, end)
    if hourly_df.empty:
        return pd.DataFrame()

    grouped = hourly_df.groupby("date").agg({
        "temperature": "mean",
        "humidity": "mean",
        "wind_speed": "mean",
        "pressure": "mean"
    }).reset_index()

    grouped["latitude"] = lat
    grouped["longitude"] = lon
    return grouped


def add_weather_columns(df: pd.DataFrame, sleep_seconds: int = 15) -> pd.DataFrame:
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    stations = df[["station_name", "latitude", "longitude"]].drop_duplicates()
    all_weather = []

    station_counter = 1
    total_stations = len(stations)

    for _, row in stations.iterrows():
        logging.info(f"Fetching weather for {row['station_name']} ({station_counter}/{total_stations})")
        station_counter += 1

        weather = fetch_daily_weather(row["latitude"], row["longitude"], start_date, end_date)
        if not weather.empty:
            weather["station_name"] = row["station_name"]
            all_weather.append(weather)
        time.sleep(sleep_seconds)

    if not all_weather:
        logging.warning("No weather data fetched.")
        return df

    weather_df = pd.concat(all_weather, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    enriched = pd.merge(
        df,
        weather_df,
        on=["date", "station_name", "latitude", "longitude"],
        how="left"
    )
    enriched["date"] = pd.to_datetime(enriched["date"])
    return enriched
