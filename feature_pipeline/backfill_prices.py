from datetime import date, timedelta

from common import config
from common.hopsworks_utils import get_feature_store, get_or_create_price_fg
from feature_pipeline.entsoe_client import fetch_prices_range
from feature_pipeline.price_features import add_price_features

BACKFILL_DAYS = 730


def main():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=BACKFILL_DAYS)
    print(f"Backfilling prices for {config.NORDPOOL_AREA} from {start} to {end}...")

    prices = fetch_prices_range(start, end)
    if prices.empty:
        raise RuntimeError("No price data fetched - check ENTSOE_API_KEY/connectivity")

    prices = add_price_features(prices)

    _, fs = get_feature_store()
    fg = get_or_create_price_fg(fs)
    fg.insert(prices)
    print(f"Inserted {len(prices)} rows into '{config.PRICE_FG_NAME}'")


if __name__ == "__main__":
    main()
