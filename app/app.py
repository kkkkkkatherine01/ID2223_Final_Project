import os
import sys
from datetime import timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.hopsworks_utils import get_feature_store, get_or_create_price_fg  # noqa: E402
from inference_pipeline.daily_inference import get_or_create_predictions_fg  # noqa: E402

st.set_page_config(page_title="SE3 Electricity Price Forecast", layout="wide")


@st.cache_resource(ttl=3600)
def get_fs():
    return get_feature_store()


@st.cache_data(ttl=900)
def load_data():
    _, fs = get_fs()
    price_df = get_or_create_price_fg(fs).read().sort_values("datetime")
    pred_df = get_or_create_predictions_fg(fs).read().sort_values("datetime")
    return price_df, pred_df


st.title("Swedish Electricity Price Forecast (SE3 - Stockholm)")

price_df, pred_df = load_data()
now = pd.Timestamp.utcnow()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Price history (7d) & 24h forecast")
    history_window = price_df[price_df["datetime"] >= now - timedelta(days=7)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_window["datetime"],
            y=history_window["price_eur_mwh"],
            name="Actual price",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df["datetime"],
            y=pred_df["predicted_price_eur_mwh"],
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="EUR / MWh",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Best times to use electricity")
    upcoming = pred_df[pred_df["datetime"] >= now].head(24)
    if upcoming.empty:
        st.info("No forecast available yet - check back after the next inference run.")
    else:
        cheapest = upcoming.nsmallest(3, "predicted_price_eur_mwh")
        for _, row in cheapest.iterrows():
            st.metric(
                label=row["datetime"].strftime("%a %H:%M"),
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

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=merged["datetime"], y=merged["price_eur_mwh"], name="Actual")
    )
    fig2.add_trace(
        go.Scatter(x=merged["datetime"], y=merged["predicted_price_eur_mwh"], name="Predicted")
    )
    fig2.update_layout(xaxis_title="Time (UTC)", yaxis_title="EUR / MWh", height=350)
    st.plotly_chart(fig2, use_container_width=True)
