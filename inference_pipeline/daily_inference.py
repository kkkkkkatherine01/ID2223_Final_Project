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


def build_inference_features(fs, target_start: datetime, target_end: datetime) -> pd.DataFrame:
    weather_fg = get_or_create_weather_fg(fs)
    weather_df = weather_fg.read()
    weather_df = weather_df[
        (weather_df["datetime"] >= target_start) & (weather_df["datetime"] <= target_end)
    ]

    price_fg = get_or_create_price_fg(fs)
    price_df = price_fg.read()
    price_cols = ["datetime"] + [f"price_lag_{lag}h" for lag in config.PRICE_LAG_HOURS] + [
        f"price_rolling_mean_{w}h" for w in config.ROLLING_WINDOWS_HOURS
    ] + [f"price_rolling_std_{w}h" for w in config.ROLLING_WINDOWS_HOURS]
    latest_price_features = price_df.sort_values("datetime")[price_cols].iloc[[-1]]

    df = weather_df.copy()
    for col in price_cols[1:]:
        df[col] = latest_price_features[col].values[0]

    return df.dropna(subset=FEATURE_COLUMNS).sort_values("datetime").reset_index(drop=True)


def main():
    project, fs = get_feature_store()

    now = datetime.now(timezone.utc)
    target_start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    target_end = target_start + timedelta(hours=config.FORECAST_HORIZON_HOURS - 1)

    features_df = build_inference_features(fs, target_start, target_end)
    if features_df.empty:
        raise RuntimeError(
            "No usable feature rows for the forecast window - make sure the "
            "feature pipeline ran today (weather forecast + price update)"
        )

    model, model_version = load_latest_model(project)
    preds = model.predict(features_df[FEATURE_COLUMNS])

    result = pd.DataFrame(
        {
            "datetime": features_df["datetime"],
            "predicted_price_eur_mwh": preds,
            "model_version": model_version,
            "prediction_made_at": now,
        }
    )

    fg = get_or_create_predictions_fg(fs)
    fg.insert(result)
    print(f"Inserted {len(result)} predictions (model v{model_version}) into '{config.PREDICTIONS_FG_NAME}'")


if __name__ == "__main__":
    main()
