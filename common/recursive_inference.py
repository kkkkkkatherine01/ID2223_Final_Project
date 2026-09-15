import numpy as np
import pandas as pd
import xgboost as xgb

from common import config


def _price_lag_features(price_series: pd.Series, t: pd.Timestamp) -> dict:
    # lag_Nh(t) for t in tomorrow's delivery day always falls on today or
    # earlier (min lag 24h == 1 day), which is already published - no
    # recursion needed here, unlike _rolling_features.
    return {f"price_lag_{lag}h": price_series.get(t - pd.Timedelta(hours=lag)) for lag in config.PRICE_LAG_HOURS}


def _rolling_features(buffer: pd.Series) -> dict:
    feats = {}
    for window in config.ROLLING_WINDOWS_HOURS:
        tail = buffer.tail(window)
        feats[f"price_rolling_mean_{window}h"] = tail.mean()
        feats[f"price_rolling_std_{window}h"] = tail.std()
    return feats


def recursive_predict(
    model: xgb.XGBRegressor, weather_df: pd.DataFrame, price_series: pd.Series, initial_buffer: pd.Series
) -> pd.DataFrame:
    buffer = initial_buffer.copy()
    preds = []
    for _, row in weather_df.iterrows():
        t = row["datetime"]
        feat = row.to_dict()
        feat.update(_price_lag_features(price_series, t))
        feat.update(_rolling_features(buffer))

        feat_row = pd.DataFrame([feat])[config.FEATURE_COLUMNS]
        if feat_row.isna().any(axis=None):
            preds.append(float("nan"))
            continue

        pred = float(model.predict(feat_row)[0])
        preds.append(pred)
        buffer.loc[t] = pred

    result = weather_df[["datetime"]].copy()
    result["predicted_price_eur_mwh"] = np.asarray(preds, dtype="float32")
    return result.dropna(subset=["predicted_price_eur_mwh"]).reset_index(drop=True)
