from datetime import date

import pandas as pd
import requests

from common import config

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_forecast() -> pd.DataFrame:
    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "hourly": "temperature_2m,wind_speed_10m,cloud_cover",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "cloud_coverage": hourly["cloud_cover"],
        }
    )
    df["cloud_coverage"] = df["cloud_coverage"].round().astype("int64")  # matches bigint FG schema
    df["source"] = "forecast"
    df["forecast_made_at"] = pd.Timestamp.now(tz="UTC")
    return df


def fetch_historical(start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,wind_speed_10m,cloud_cover",
        "timezone": "UTC",
        "wind_speed_unit": "ms",  # Open-Meteo defaults to km/h - pin explicitly, don't rely on it
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "cloud_coverage": hourly["cloud_cover"],
        }
    )
    df["source"] = "historical_archive"
    df["forecast_made_at"] = df["datetime"]
    return df
