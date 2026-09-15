from datetime import datetime, timedelta, timezone

import pandas as pd
import xgboost as xgb

from common import config
from common.hopsworks_utils import (
    get_champion_model,
    get_feature_store,
    get_or_create_predictions_fg,
    get_or_create_price_fg,
    get_or_create_weather_fg,
)
from common.recursive_inference import predict_next_24h


def load_latest_model(project):
    hw_model = get_champion_model(project)
    if hw_model is None:
        raise RuntimeError(f"No model registered for '{config.MODEL_NAME}' yet - run training_pipeline first")
    model_dir = hw_model.download()
    model = xgb.XGBRegressor()
    model.load_model(f"{model_dir}/model.json")
    return model, hw_model.version


def _assert_complete_hourly_coverage(df: pd.DataFrame, target_start: datetime, target_end: datetime, label: str):
    expected = pd.date_range(target_start, target_end, freq="1h", tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(df["datetime"]))
    if len(missing) > 0:
        raise RuntimeError(
            f"{label}: missing {len(missing)}/{len(expected)} required hours for the "
            f"{config.FORECAST_HORIZON_HOURS}h forecast window: {list(missing)}"
        )


def build_inference_inputs(fs, target_start: datetime, target_end: datetime):
    weather_fg = get_or_create_weather_fg(fs)
    weather_df = weather_fg.read()
    weather_df = weather_df[
        (weather_df["datetime"] >= target_start) & (weather_df["datetime"] <= target_end)
    ].sort_values("datetime").reset_index(drop=True)
    _assert_complete_hourly_coverage(weather_df, target_start, target_end, "weather forecast")

    price_fg = get_or_create_price_fg(fs)
    price_df = price_fg.read().sort_values("datetime")
    price_series = price_df.set_index("datetime")["price_eur_mwh"]

    max_window = max(config.ROLLING_WINDOWS_HOURS)
    initial_buffer = price_series[price_series.index < target_start].tail(max_window)

    return weather_df, price_series, initial_buffer


def main():
    project, fs = get_feature_store()

    now = datetime.now(timezone.utc)
    target_start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    target_end = target_start + timedelta(hours=config.FORECAST_HORIZON_HOURS - 1)

    weather_df, price_series, initial_buffer = build_inference_inputs(fs, target_start, target_end)

    model, model_version = load_latest_model(project)
    result = predict_next_24h(model, weather_df, price_series, initial_buffer)
    _assert_complete_hourly_coverage(result, target_start, target_end, "predictions")

    result["model_version"] = model_version
    result["prediction_made_at"] = now

    fg = get_or_create_predictions_fg(fs)
    fg.insert(result)
    print(f"Inserted {len(result)} predictions (model v{model_version}) into '{config.PREDICTIONS_FG_NAME}'")


if __name__ == "__main__":
    main()
