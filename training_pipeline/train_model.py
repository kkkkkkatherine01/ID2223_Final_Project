import json
import os
from datetime import timedelta

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common import config
from common.hopsworks_utils import get_feature_store, get_or_create_feature_view

TEST_DAYS = 30
VAL_DAYS = 21

FEATURE_COLUMNS = [
    "temperature",
    "wind_speed",
    "cloud_coverage",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "month",
] + [f"price_lag_{lag}h" for lag in config.PRICE_LAG_HOURS] + [
    f"price_rolling_mean_{w}h" for w in config.ROLLING_WINDOWS_HOURS
] + [
    f"price_rolling_std_{w}h" for w in config.ROLLING_WINDOWS_HOURS
]
TARGET_COLUMN = "price_eur_mwh"


def load_training_data(fs) -> pd.DataFrame:
    fv = get_or_create_feature_view(fs)
    df = fv.get_batch_data()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df.sort_values("datetime").reset_index(drop=True)


def time_split(df: pd.DataFrame):
    cutoff = df["datetime"].max() - timedelta(days=TEST_DAYS)
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]
    return train, test


def train(train_df: pd.DataFrame) -> xgb.XGBRegressor:
    val_cutoff = train_df["datetime"].max() - timedelta(days=VAL_DAYS)
    fit_df = train_df[train_df["datetime"] < val_cutoff]
    val_df = train_df[train_df["datetime"] >= val_cutoff]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        early_stopping_rounds=30,
    )
    model.fit(
        fit_df[FEATURE_COLUMNS],
        fit_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def evaluate(model: xgb.XGBRegressor, test_df: pd.DataFrame) -> dict:
    preds = model.predict(test_df[FEATURE_COLUMNS])
    return {
        "mae": float(mean_absolute_error(test_df[TARGET_COLUMN], preds)),
        "rmse": float(mean_squared_error(test_df[TARGET_COLUMN], preds) ** 0.5),
        "n_test_rows": int(len(test_df)),
    }


def main():
    project, fs = get_feature_store()
    df = load_training_data(fs)
    if len(df) < 24 * 60:
        raise RuntimeError(f"Only {len(df)} usable rows - backfill more history before training")

    train_df, test_df = time_split(df)
    model = train(train_df)
    metrics = evaluate(model, test_df)
    print("Evaluation:", metrics)

    out_dir = "model_artifact"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.json")
    model.save_model(model_path)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=config.MODEL_NAME,
        metrics=metrics,
        description="XGBoost regressor for SE3 hourly day-ahead electricity price",
        input_example=train_df[FEATURE_COLUMNS].iloc[:1],
    )
    hw_model.save(out_dir)
    print(f"Registered model '{config.MODEL_NAME}' version {hw_model.version}")


if __name__ == "__main__":
    main()
