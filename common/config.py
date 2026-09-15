import os
from dotenv import load_dotenv

load_dotenv()

NORDPOOL_AREA = os.getenv("NORDPOOL_AREA", "SE3")
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

WEATHER_LAT = float(os.getenv("WEATHER_LAT", "59.3293"))
WEATHER_LON = float(os.getenv("WEATHER_LON", "18.0686"))

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FG_VERSION = 1
PRICE_FG_NAME = "electricity_prices"
WEATHER_FG_NAME = "weather"
PREDICTIONS_FG_NAME = "price_predictions"
PREDICTIONS_FG_VERSION = 3

FEATURE_VIEW_NAME = "electricity_price_fv"
FEATURE_VIEW_VERSION = 2

MODEL_NAME = "electricity_price_xgb"
MODEL_VERSION = None

PRICE_LAG_HOURS = [24, 48, 168]
ROLLING_WINDOWS_HOURS = [24, 168]

STOCKHOLM_TZ = "Europe/Stockholm"
# backtest simulates daily_inference.py running at this UTC hour each day
INFERENCE_RUN_HOUR_UTC = 8

FEATURE_COLUMNS = [
    "temperature",
    "wind_speed",
    "cloud_coverage",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "month",
] + [f"price_lag_{lag}h" for lag in PRICE_LAG_HOURS] + [
    f"price_rolling_mean_{w}h" for w in ROLLING_WINDOWS_HOURS
] + [
    f"price_rolling_std_{w}h" for w in ROLLING_WINDOWS_HOURS
]
TARGET_COLUMN = "price_eur_mwh"
