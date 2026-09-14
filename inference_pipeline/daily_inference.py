from datetime import datetime, timedelta, timezone

import pandas as pd
import xgboost as xgb

from common import config
from common.hopsworks_utils import (
    get_feature_store,
    get_or_create_predictions_fg,
    get_or_create_price_fg,
    get_or_create_weather_fg,
)
from training_pipeline.train_model import FEATURE_COLUMNS


def load_latest_model(project):
    mr = project.get_model_registry()
    hw_model = mr.get_best_model(config.MODEL_NAME, "mae", "min")
    model_dir = hw_model.download()
    model = xgb.XGBRegressor()
    model.load_model(f"{model_dir}/model.json")
    return model, hw_model.version


def _price_lag_features(price_series: pd.Series, t: pd.Timestamp) -> dict:
    # lag_Nh for a target hour t is price(t - Nh); since the smallest lag (24h)
    # is >= the forecast horizon, t - Nh is always in the past and observed -
    # a direct lookup, never an approximation.
    return {f"price_lag_{lag}h": price_series.get(t - pd.Timedelta(hours=lag)) for lag in config.PRICE_LAG_HOURS}


def _rolling_features(buffer: pd.Series) -> dict:
    feats = {}
    for window in config.ROLLING_WINDOWS_HOURS:
        tail = buffer.tail(window)
        feats[f"price_rolling_mean_{window}h"] = tail.mean()
        feats[f"price_rolling_std_{window}h"] = tail.std()
    return feats


def build_inference_inputs(fs, target_start: datetime, target_end: datetime):
    """Fetch the target window's weather/calendar rows plus raw price history.

    The price history is used two ways: exact point lookups for the lag_Nh
    features (always safe - see _price_lag_features), and as the starting
    buffer for the rolling_Wh features, which predict_next_24h extends
    hour-by-hour with each step's own prediction.
    """
    weather_fg = get_or_create_weather_fg(fs)
    weather_df = weather_fg.read()
    weather_df = weather_df[
        (weather_df["datetime"] >= target_start) & (weather_df["datetime"] <= target_end)
    ].sort_values("datetime").reset_index(drop=True)

    price_fg = get_or_create_price_fg(fs)
    price_df = price_fg.read().sort_values("datetime")
    price_series = price_df.set_index("datetime")["price_eur_mwh"]

    max_window = max(config.ROLLING_WINDOWS_HOURS)
    initial_buffer = price_series[price_series.index < target_start].tail(max_window)

    return weather_df, price_series, initial_buffer


def predict_next_24h(
    model: xgb.XGBRegressor, weather_df: pd.DataFrame, price_series: pd.Series, initial_buffer: pd.Series
) -> pd.DataFrame:
    """Recursive multi-step forecast: rolling_Wh features can reach into hours
    that haven't happened yet (the 24h rolling window vs. the 24h forecast
    horizon), so each hour's own prediction is fed back into the buffer as a
    stand-in "observed" price for computing the next hour's rolling features -
    the standard recursive strategy for multi-step forecasting (same pattern
    used for pm25_lag_* in the course's air-quality inference notebook).
    """
    buffer = initial_buffer.copy()
    preds = []
    for _, row in weather_df.iterrows():
        t = row["datetime"]
        feat = row.to_dict()
        feat.update(_price_lag_features(price_series, t))
        feat.update(_rolling_features(buffer))

        feat_row = pd.DataFrame([feat])[FEATURE_COLUMNS]
        if feat_row.isna().any(axis=None):
            preds.append(float("nan"))
            continue

        pred = float(model.predict(feat_row)[0])
        preds.append(pred)
        buffer.loc[t] = pred

    result = weather_df[["datetime"]].copy()
    result["predicted_price_eur_mwh"] = preds
    return result.dropna(subset=["predicted_price_eur_mwh"]).reset_index(drop=True)


def main():
    project, fs = get_feature_store()

    now = datetime.now(timezone.utc)
    target_start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    target_end = target_start + timedelta(hours=config.FORECAST_HORIZON_HOURS - 1)

    weather_df, price_series, initial_buffer = build_inference_inputs(fs, target_start, target_end)
    if weather_df.empty:
        raise RuntimeError(
            "No usable feature rows for the forecast window - make sure the "
            "feature pipeline ran today (weather forecast + price update)"
        )

    model, model_version = load_latest_model(project)
    result = predict_next_24h(model, weather_df, price_series, initial_buffer)
    if result.empty:
        raise RuntimeError(
            "No predictions produced for the forecast window - check weather/price feature coverage"
        )

    result["model_version"] = model_version
    result["prediction_made_at"] = now

    fg = get_or_create_predictions_fg(fs)
    fg.insert(result)
    print(f"Inserted {len(result)} predictions (model v{model_version}) into '{config.PREDICTIONS_FG_NAME}'")


if __name__ == "__main__":
    main()
