import pandas as pd
import holidays


def build_calendar_features(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    df = pd.DataFrame({"datetime": idx})
    return add_calendar_features(df)


def add_calendar_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    se_holidays = holidays.Sweden()
    dt = df[datetime_col]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.dt.date.map(lambda d: d in se_holidays).astype(int)
    df["month"] = dt.dt.month
    return df
