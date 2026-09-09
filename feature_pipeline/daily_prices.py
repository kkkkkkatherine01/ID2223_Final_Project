from datetime import date, timedelta

from common import config
from common.hopsworks_utils import get_feature_store, get_or_create_price_fg
from feature_pipeline.entsoe_client import fetch_prices_range
from feature_pipeline.price_features import add_price_features

CONTEXT_DAYS = 14
INSERT_DAYS = 3
FORWARD_DAYS = 1


def main():
    end = date.today() + timedelta(days=FORWARD_DAYS)
    context_start = date.today() - timedelta(days=CONTEXT_DAYS)
    print(f"Fetching prices for {config.NORDPOOL_AREA} from {context_start} to {end}...")

    prices = fetch_prices_range(context_start, end)
    if prices.empty:
        raise RuntimeError("No price data fetched today - check ENTSOE_API_KEY/connectivity")

    prices = add_price_features(prices)

    insert_cutoff = date.today() - timedelta(days=INSERT_DAYS)
    prices = prices[prices["datetime"].dt.date >= insert_cutoff]

    _, fs = get_feature_store()
    fg = get_or_create_price_fg(fs)
    fg.insert(prices)
    print(f"Upserted {len(prices)} rows into '{config.PRICE_FG_NAME}'")


if __name__ == "__main__":
    main()
