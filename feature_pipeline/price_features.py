import pandas as pd

from common import config


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    area = df["area"].iloc[0] if len(df) else config.NORDPOOL_AREA

    df = df.set_index("datetime")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full_index)
    df["price_eur_mwh"] = df["price_eur_mwh"].ffill(limit=6)

    for lag in config.PRICE_LAG_HOURS:
        df[f"price_lag_{lag}h"] = df["price_eur_mwh"].shift(lag)
    for window in config.ROLLING_WINDOWS_HOURS:
        shifted = df["price_eur_mwh"].shift(1)
        df[f"price_rolling_mean_{window}h"] = shifted.rolling(window).mean()
        df[f"price_rolling_std_{window}h"] = shifted.rolling(window).std()

    df = df.rename_axis("datetime").reset_index()
    df["area"] = area
    return df
