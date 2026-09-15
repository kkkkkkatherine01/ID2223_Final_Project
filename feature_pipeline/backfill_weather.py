from datetime import datetime, timedelta, timezone

from common import config
from common.calendar_features import add_calendar_features
from common.hopsworks_utils import get_feature_store, get_or_create_weather_fg
from feature_pipeline.weather_client import fetch_historical

BACKFILL_DAYS = 730


def main():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=BACKFILL_DAYS)
    print(f"Backfilling weather from {start} to {end}...")

    weather = fetch_historical(start, end)
    if weather.empty:
        raise RuntimeError("No weather data fetched - check Open-Meteo connectivity")
    weather = add_calendar_features(weather)

    _, fs = get_feature_store()
    fg = get_or_create_weather_fg(fs)
    fg.insert(weather)
    print(f"Inserted {len(weather)} rows into '{config.WEATHER_FG_NAME}'")


if __name__ == "__main__":
    main()
