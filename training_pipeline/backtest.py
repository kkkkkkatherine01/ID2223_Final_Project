import pandas as pd
import xgboost as xgb

from common import config
from common.recursive_inference import predict_next_24h

HORIZON = config.FORECAST_HORIZON_HOURS
_WEATHER_CALENDAR_COLS = [c for c in config.FEATURE_COLUMNS if not c.startswith("price_")]


def recursive_backtest_mae(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    price_series_raw: pd.Series,
    test_days: int,
    step_hours: int = 24,
) -> dict:
    d = df.set_index("datetime").sort_index()
    full_index = pd.date_range(d.index.min(), d.index.max(), freq="1h", tz=d.index.tz)
    d = d.reindex(full_index)
    d.index.name = "datetime"

    price_series_full = price_series_raw.sort_index()
    max_window = max(config.ROLLING_WINDOWS_HOURS)

    cutoff = d.index.max() - pd.Timedelta(days=test_days)
    idx = d.index[d.index >= cutoff]

    abs_errors = []
    for i in range(0, len(idx) - HORIZON, step_hours):
        T = idx[i]
        window = d.loc[T : T + pd.Timedelta(hours=HORIZON)]
        if len(window) != HORIZON + 1:
            continue
        if window[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].isna().any().any():
            continue

        target_rows = window.iloc[1:]
        weather_df = target_rows.reset_index()[["datetime"] + _WEATHER_CALENDAR_COLS]
        initial_buffer = price_series_full[price_series_full.index < T + pd.Timedelta(hours=1)].tail(max_window)

        pred = predict_next_24h(model, weather_df, price_series_full, initial_buffer)
        if len(pred) != HORIZON:
            continue
        pred = pred.set_index("datetime")

        abs_err = (target_rows[config.TARGET_COLUMN] - pred["predicted_price_eur_mwh"]).abs()
        abs_errors.extend(abs_err.tolist())

    if not abs_errors:
        return {"mae": float("nan"), "n_windows": 0}

    return {
        "mae": float(pd.Series(abs_errors).mean()),
        "n_windows": len(abs_errors) // HORIZON,
    }
