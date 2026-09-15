import json
import math
import os
from datetime import timedelta

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common import config
from common.config import FEATURE_COLUMNS, TARGET_COLUMN
from common.hopsworks_utils import (
    get_champion_model,
    get_feature_store,
    get_or_create_feature_view,
    get_or_create_price_fg,
)
from training_pipeline.backtest import recursive_backtest_mae

TEST_DAYS = 30
VAL_DAYS = 21


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

    price_series_raw = get_or_create_price_fg(fs).read().set_index("datetime")["price_eur_mwh"].sort_index()

    train_df, test_df = time_split(df)
    model = train(train_df)

    oracle_metrics = evaluate(model, test_df)

    candidate_backtest = recursive_backtest_mae(model, df, price_series_raw, test_days=TEST_DAYS)
    if candidate_backtest["n_windows"] == 0 or not math.isfinite(candidate_backtest["mae"]):
        raise RuntimeError(
            f"Candidate backtest produced no valid evaluation windows "
            f"(n_windows={candidate_backtest['n_windows']}, mae={candidate_backtest['mae']}) "
            "- refusing to register an unevaluated model"
        )
    metrics = {
        "mae": candidate_backtest["mae"],
        "oracle_mae": oracle_metrics["mae"],
        "rmse": oracle_metrics["rmse"],
        "n_test_rows": oracle_metrics["n_test_rows"],
        "n_backtest_windows": candidate_backtest["n_windows"],
    }
    print("Evaluation:", metrics)

    mr = project.get_model_registry()

    champion_mae = None
    champion = get_champion_model(project)
    if champion is None:
        print("No existing champion to compare against; registering unconditionally.")
    else:
        champion_dir = champion.download()
        champion_model = xgb.XGBRegressor()
        champion_model.load_model(f"{champion_dir}/model.json")
        champion_backtest = recursive_backtest_mae(champion_model, df, price_series_raw, test_days=TEST_DAYS)
        if champion_backtest["n_windows"] == 0 or not math.isfinite(champion_backtest["mae"]):
            print(
                f"Champion (v{champion.version})'s backtest produced no valid evaluation windows on this "
                "window - treating the candidate as automatically better."
            )
        else:
            champion_mae = champion_backtest["mae"]
            print(f"Current champion (v{champion.version}) re-evaluated on this window: mae={champion_mae:.3f}")

    if champion_mae is not None and metrics["mae"] >= champion_mae:
        print(
            f"Candidate mae={metrics['mae']:.3f} is not better than champion's "
            f"mae={champion_mae:.3f} on the same window - skipping registration."
        )
        return

    out_dir = "model_artifact"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.json")
    model.save_model(model_path)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

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
