import time

import pandas as pd
import requests
from entsoe import EntsoePandasClient

from common import config

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 30

# ENTSO-E area codes for the Swedish bidding zones, keyed by config.NORDPOOL_AREA
AREA_CODES = {
    "SE1": "SE_1",
    "SE2": "SE_2",
    "SE3": "SE_3",
    "SE4": "SE_4",
}


def _area_code() -> str:
    try:
        return AREA_CODES[config.NORDPOOL_AREA]
    except KeyError:
        raise RuntimeError(
            f"Unknown NORDPOOL_AREA '{config.NORDPOOL_AREA}' - must be one of {sorted(AREA_CODES)}"
        )


def _client() -> EntsoePandasClient:
    if not config.ENTSOE_API_KEY:
        raise RuntimeError("ENTSOE_API_KEY not set - see .env.example")
    return EntsoePandasClient(api_key=config.ENTSOE_API_KEY)


def _to_local(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("Europe/Stockholm") if ts.tzinfo is None else ts.tz_convert("Europe/Stockholm")


def fetch_prices_range(start, end) -> pd.DataFrame:
    client = _client()

    area_code = _area_code()
    series = None
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            series = client.query_day_ahead_prices(
                area_code, start=_to_local(start), end=_to_local(end)
            )
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            print(f"[entsoe] attempt {attempt + 1}/{MAX_RETRIES} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    if series is None:
        raise last_exc

    df = series.rename("price_eur_mwh").to_frame()
    df.index = df.index.tz_convert("UTC")
    df = df.resample("1h").mean()  # SE3 is 15-min resolution since 2025
    df = df.rename_axis("datetime").reset_index()
    df["area"] = config.NORDPOOL_AREA
    return df[["datetime", "area", "price_eur_mwh"]]
