import os
import sys
from datetime import timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.hopsworks_utils import (  # noqa: E402
    get_feature_store,
    get_or_create_predictions_fg,
    get_or_create_price_fg,
)

st.set_page_config(page_title="SE3 Electricity Price Forecast", layout="wide")

STOCKHOLM_TZ = "Europe/Stockholm"


@st.cache_resource(ttl=3600)
def get_fs():
    return get_feature_store()


@st.cache_data(ttl=900)
def load_data():
    _, fs = get_fs()
    price_df = get_or_create_price_fg(fs).read().sort_values("datetime")
    pred_df = get_or_create_predictions_fg(fs).read()
    pred_df = pred_df.sort_values("prediction_made_at").drop_duplicates(subset="datetime", keep="last")
    pred_df = pred_df.sort_values("datetime")
    return price_df, pred_df


st.title("Swedish Electricity Price Forecast (SE3 - Stockholm)")

price_df, pred_df = load_data()
now = pd.Timestamp.utcnow()

latest_batch = pred_df["prediction_made_at"].max() if not pred_df.empty else None
forecast_df = pred_df[pred_df["prediction_made_at"] == latest_batch] if latest_batch is not None else pred_df.iloc[0:0]
forecast_df = forecast_df.sort_values("datetime")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Forecast")
    if forecast_df.empty:
        st.info("No forecast available yet - check back after the next inference run.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=forecast_df["datetime"].dt.tz_convert(STOCKHOLM_TZ),
                y=forecast_df["predicted_price_eur_mwh"],
                name="Forecast",
                line=dict(color="#ff7f0e"),
            )
        )
        fig.update_layout(
            xaxis_title="Time (Stockholm)",
            yaxis_title="EUR / MWh",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Best times to use electricity")
    known_upcoming = price_df[price_df["datetime"] > now][["datetime", "price_eur_mwh"]]
    known_upcoming = known_upcoming.rename(columns={"price_eur_mwh": "predicted_price_eur_mwh"})
    upcoming_forecast = forecast_df[forecast_df["datetime"] > now][["datetime", "predicted_price_eur_mwh"]]
    upcoming = pd.concat([known_upcoming, upcoming_forecast], ignore_index=True)
    upcoming = upcoming.drop_duplicates(subset="datetime", keep="first")
    if upcoming.empty:
        st.info("No forecast available yet - check back after the next inference run.")
    else:
        cheapest = upcoming.nsmallest(3, "predicted_price_eur_mwh")
        for _, row in cheapest.iterrows():
            local_time = row["datetime"].tz_convert(STOCKHOLM_TZ)
            st.metric(
                label=f"{local_time.strftime('%a %H:%M %Z')} (Stockholm)",
                value=f"{row['predicted_price_eur_mwh']:.1f} EUR/MWh",
            )

st.subheader("Predicted vs actual (last 7 days)")
merged = pd.merge(
    pred_df[pred_df["datetime"] < now],
    price_df[["datetime", "price_eur_mwh"]],
    on="datetime",
    how="inner",
).sort_values("datetime")
merged = merged[merged["datetime"] >= now - timedelta(days=7)]

if merged.empty:
    st.info("Not enough overlapping history yet to compare predictions vs actuals.")
else:
    mae = (merged["predicted_price_eur_mwh"] - merged["price_eur_mwh"]).abs().mean()
    st.caption(f"MAE over shown window: {mae:.2f} EUR/MWh")

    merged_local = merged["datetime"].dt.tz_convert(STOCKHOLM_TZ)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=merged_local, y=merged["price_eur_mwh"], name="Actual")
    )
    fig2.add_trace(
        go.Scatter(x=merged_local, y=merged["predicted_price_eur_mwh"], name="Predicted")
    )
    fig2.update_layout(xaxis_title="Time (Stockholm)", yaxis_title="EUR / MWh", height=350)
    st.plotly_chart(fig2, use_container_width=True)
