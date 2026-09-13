import os
from dotenv import load_dotenv

load_dotenv()

NORDPOOL_AREA = os.getenv("NORDPOOL_AREA", "SE3")
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHER_LAT = float(os.getenv("WEATHER_LAT", "59.3293"))
WEATHER_LON = float(os.getenv("WEATHER_LON", "18.0686"))

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FG_VERSION = 1
PRICE_FG_NAME = "electricity_prices"
WEATHER_FG_NAME = "weather"
PREDICTIONS_FG_NAME = "price_predictions"

FEATURE_VIEW_NAME = "electricity_price_fv"
FEATURE_VIEW_VERSION = 2

MODEL_NAME = "electricity_price_xgb"
MODEL_VERSION = None

PRICE_LAG_HOURS = [24, 48, 168]
ROLLING_WINDOWS_HOURS = [24, 168]
FORECAST_HORIZON_HOURS = 24
