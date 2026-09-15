from datetime import date

import pandas as pd
import requests

from common import config

OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_forecast() -> pd.DataFrame:
    params = {
        "lat": config.WEATHER_LAT,
        "lon": config.WEATHER_LON,
        "appid": config.OPENWEATHER_API_KEY,
        "units": "metric",
    }
    resp = requests.get(OWM_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = [
        {
            "datetime": pd.to_datetime(item["dt"], unit="s", utc=True),
            "temperature": item["main"]["temp"],
            "wind_speed": item["wind"]["speed"],
            "cloud_coverage": item["clouds"]["all"],
        }
        for item in payload["list"]
    ]
    df = pd.DataFrame(rows).sort_values("datetime").set_index("datetime")
    df = df.resample("1h").interpolate(method="linear").reset_index()
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
        "wind_speed_unit": "ms",  # Open-Meteo defaults to km/h; forecast (OWM, units=metric) is m/s
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
