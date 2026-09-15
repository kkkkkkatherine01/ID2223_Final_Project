import pandas as pd
import xgboost as xgb

from common import config
from common.delivery_day import next_delivery_day_window
from common.recursive_inference import recursive_predict

_WEATHER_CALENDAR_COLS = [c for c in config.FEATURE_COLUMNS if not c.startswith("price_")]


def recursive_backtest_mae(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    price_series_raw: pd.Series,
    test_days: int,
    step_days: int = 1,
) -> dict:
    """Simulates daily_inference.py running once a day at config.INFERENCE_RUN_HOUR_UTC,
    predicting the next Stockholm delivery day, over the last `test_days`.
    """
    d = df.set_index("datetime").sort_index()
    full_index = pd.date_range(d.index.min(), d.index.max(), freq="1h", tz=d.index.tz)
    d = d.reindex(full_index)
    d.index.name = "datetime"

    price_series_full = price_series_raw.sort_index()
    max_window = max(config.ROLLING_WINDOWS_HOURS)

    sim_date = pd.Timestamp(d.index.max().date() - pd.Timedelta(days=test_days), tz=d.index.tz)
    end_date = pd.Timestamp(d.index.max().date(), tz=d.index.tz)

    abs_errors = []
    n_windows = 0
    while sim_date <= end_date:
        reference_time = sim_date + pd.Timedelta(hours=config.INFERENCE_RUN_HOUR_UTC)
        target_start, target_end = next_delivery_day_window(reference_time)
        target_start = pd.Timestamp(target_start).tz_convert(d.index.tz)
        target_end = pd.Timestamp(target_end).tz_convert(d.index.tz)

        window = d.loc[target_start:target_end]
        expected_len = len(pd.date_range(target_start, target_end, freq="1h", tz=d.index.tz))
        sim_date += pd.Timedelta(days=step_days)

        if len(window) != expected_len:
            continue
        if window[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].isna().any().any():
            continue

        weather_df = window.reset_index()[["datetime"] + _WEATHER_CALENDAR_COLS]
        initial_buffer = price_series_full[price_series_full.index < target_start].tail(max_window)

        pred = recursive_predict(model, weather_df, price_series_full, initial_buffer)
        if len(pred) != expected_len:
            continue
        pred = pred.set_index("datetime")

        abs_err = (window[config.TARGET_COLUMN] - pred["predicted_price_eur_mwh"]).abs()
        abs_errors.extend(abs_err.tolist())
        n_windows += 1

    if not abs_errors:
        return {"mae": float("nan"), "n_windows": 0}

    return {
        "mae": float(pd.Series(abs_errors).mean()),
        "n_windows": n_windows,
    }
